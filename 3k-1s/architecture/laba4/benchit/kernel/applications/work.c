/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures read latency of data located in different cache
 *         levels or memory of certain CPUs.
 *******************************************************************/

/*
 * TODO - update page selection to handle hugepages differently
 *      - TLB optimization for hugepages (LLC size might soon exceed 2M entries)
 *      - adopt cache and TLB parameters to refer to identifiers returned by 
 *        the hardware detection
 *      - AVX and Larrabee support
 *      - support low level Performance Counter APIs to get access to uncore/NB events
 *      - optionally remove data from SHARED_CPU to bring data into S/F/O without a 
 *        second copy in another CPU
 *      - option for per thread allocation of flush buffer
 */
 
#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
#include "work.h"
#include "interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <pthread.h>

#include <assert.h>
#include <unistd.h>

#ifdef USE_PAPI
#include <papi.h>
#endif

/**
 * RANDOMIZE_1 | RANDOMIZE_2 | RANDOMIZE_3
 * - influences prefetcher effectiveness, default RANDOMIZE_3
 * 1: generating one random pattern per memory size used by all threads in all iterations
 * 2: Thread 0 generates one pattern for each thread and each iteration
 * 3: each thread generates its own random access pattern for each iteration
 */
#define RANDOMIZE_3

//#define AVERAGE

/** user defined maximum value of random numbers returned by _random() */
static unsigned long long random_max=0;

/** parameters for random number generator 
 *  formula: random_value(n+1) = (rand_a*random_value(n)+rand_b)%rand_m
 *  rand_fix: rand_fix=(rand_a*rand_fix+rand_b)%rand_m
 *        - won't be used as start_value
 *        - can't be reached by random_value, however special care is taken that rand_fix will also be returned by _random()
 */
static unsigned long long random_value=0;
static unsigned long long rand_a=0;
static unsigned long long rand_b=0;
static unsigned long long rand_m=1;
static unsigned long long rand_fix=0;

/** table of prime numbers needed to generate parameters for random number generator */
int *p_list=NULL;
int p_list_max=0;
int pos=0;

/** variables for prime factorization needed to generate parameters for random number generator */
long long parts [64];
int part_count;
long long number;
int max_factor;

/**
 * checks if value is prime
 * has to be called with all prime numbers < sqrt(value)+1 prior to the call with value
 */
static int isprime(unsigned long long value)
{
  int i;
  int limit = (int) trunc(sqrt((double) value)) +1;
  for (i=0;i<=pos;i++){
      if (p_list[i]>limit) break;
      if (value==(unsigned long long)p_list[i]) return 1;
      if (value%p_list[i]==0) return 0;
  }
  if (pos < p_list_max -1){
     pos++;
     p_list[pos]=value;
  }
  else
   if (p_list[pos]<limit) 
      for (i=p_list[pos];i<=limit;i+=2){
        if (value%i==0) return 0;
      }
  return 1;
}

int iteration,alignment,accesses;

/**
 * checks if value is a prime factor of global variable number
 * has to be called with all prime numbers < sqrt(value)+1 prior to the call with value
 */
static int isfactor(int value)
{
  if (value<p_list[p_list_max-1]) if (!isprime(value)) return 0;
  if (number%value==0){
     parts[part_count]=value;
     while (number%value==0){
       number=number/value;
     }
     part_count++;
     max_factor = (int) trunc(sqrt((double) number))+1;
  }
  return 1;
}

/**
 * calculates (x^y)%m
 */
unsigned long long potenz(long long x, long long y, long long m)
{
   unsigned long long res  =1;
   unsigned long long mask =1;

   if (y==0) return 1;
   if (y==1) return x%m;

   assert(y==y&0x00000000ffffffffULL);
   assert(x==x&0x00000000ffffffffULL);
   assert(m==m&0x00000000ffffffffULL);
   
   mask = mask<<63;

   while (y&mask==0) mask= mask >> 1;
   do{
        if (y&mask){
            res=(res*x)%m;
            res=(res*res)%m;
        }
        else{
            res=(res*res)%m;
        }
        mask = mask >> 1;
   }
   while (mask>1);

   if (y&mask){
      res=(res*x)%m;
   }

   return res;
}

/**
 * checks if value is a primitive root of rand_m
 */
static int isprimitiveroot(long long value)
{
  long long i,x,y;
  for (i=0;i<part_count;i++){
      x = value;
      y = (rand_m-1)/parts[i];     
      if (potenz(x,y,rand_m)==1) return 0;
  }
  return 1;
}

/**  
 * returns a pseudo random number
 * do not use this function without a prior call to _random_init()
 */
static unsigned long long _random(void)
{
  if (random_max==0) return -1;
  do{
    random_value = (random_value * rand_a + rand_b)%rand_m;
  }
  while (((random_value>random_max)&&(rand_fix<random_max))||((random_value>=random_max)&&(rand_fix>=random_max)));
  if (random_value<rand_fix) return random_value;
  else return random_value-1;
}

/**
 * Initializes the random number generator with the values given to the function.
 * formula: r(n+1) = (a*r(n)+b)%m
 * sequence generated by calls of _random() is a permutation of values from 0 to max-1
 */
static void _random_init(int start,int max)
{
  int i;
  unsigned long long x,f1,f2;

  random_max = (unsigned long long) max;
  if (random_max==0) return;
  /* allocate memory for prime number table */
  if ((((int) trunc(sqrt((double) random_max)) +1)/2+1)>p_list_max){
    p_list_max=((int) trunc(sqrt((double) random_max)) +1)/2+1;
    p_list=realloc(p_list,p_list_max*sizeof(int));
    if (p_list==NULL){
      while(p_list==NULL){
        p_list_max=p_list_max/2;
        p_list=calloc(p_list_max,sizeof(int));
        assert(p_list_max>2);
      }
      pos=0;
    }
    if (pos==0){
      p_list[0]=2;
      p_list[1]=3;
      pos++;
    }
  }

  rand_m=1;
  rand_a=0;
  /* setup parameters rand_m, rand_a, rand_b, and rand_fix*/
  do{
    rand_m+=2;
    /* find a prime number for rand_m, larger than random_max*/
    while ((pos<p_list_max-1)){rand_m+=2;isprime(rand_m);} /* fill prime number table */
    if (rand_m<=random_max) {rand_m=random_max+1;if(rand_m%2==0)rand_m++;}
    while (!isprime(rand_m)) rand_m+=2;
  
    /* set rand_b ta a value between rand_m/4 and 3*rand_m/4 */
    rand_b=(rand_m/4+start)%(3*rand_m/4);
  
    /* prime factorize rand_m-1, as those are good candidates for primitive roots of rand_m */
    number=rand_m-1;
    max_factor = (int) trunc(sqrt((double) number))+1;
    part_count=0;
    for(i=0;i<p_list_max;i++) isfactor(p_list[i]);
    i=p_list[p_list_max-1];
    while (i<max_factor){
       isfactor(i);
       i+=2;
    }
    if (number>1){
       parts[part_count]=number;
       part_count++;
    }
  
    /* 
     * find a value for rand_a that is a primitive root of rand_m and != rand_m/2 
     * rand_a = rand_m/2 has a high likelyhood to generate a regular pattern
     */
    for (i=0;i<part_count;i++){
      if ((rand_m/2!=parts[i])&&(parts[i]*parts[i]>rand_m)&&(isprimitiveroot(parts[i]))) {rand_a=parts[i];break;}
    }
    
    /* find fixpoint */
    rand_fix=0;
    if (rand_a!=0) for(x=0;x<=rand_a;x++){
        f1 = ((x*rand_m) -rand_b ) / (rand_a-1);
        f2 = ((f1*rand_a)+rand_b) % rand_m;
        if (f1==f2) {rand_fix=f1;break;}
    }    
  }
  while((rand_a*rand_a<rand_m)||(rand_fix==0));

  /*printf("rand_max: %5llu (start: %8i) - ",random_max, start);
  printf("rand_m: %5llu ",rand_m);
  printf("rand_a: %5llu ",rand_a);
  printf("rand_b: %5llu" ,rand_b);
  printf("rand_fix: %5llu\n",rand_fix);*/

  /* generator is initialized with the user defined start value */
  random_value= (unsigned long long)start%rand_m;
  if (random_value==rand_fix) random_value=0;  /* Fixpoint can't be used */
}

static void reset_tlb_check(volatile mydata_t* data)
{
  int i,j;
  
  if (data->tlb_sets) for (i=0;i<data->tlb_size/data->tlb_sets;i++)
  {
    data->tlb_collision_check_array[i]=0;
    for (j=0;j<data->tlb_sets;j++) data->tlb_tags[j*(data->tlb_size/data->tlb_sets)+i]=(unsigned long long)0;
  }
}

//checks if page-addresses fit into the n-way associative TLB
static int tlb_check(unsigned long long addr,volatile mydata_t* data)
{
  int i,indizes,index_mask,index,tmp;

  if ((data->tlb_size==0)||(data->tlb_sets==0)) return 0;
 
  indizes = data->tlb_size/data->tlb_sets;
  index_mask=indizes-1;
  tmp=addr/data->pagesize;
  index=tmp&index_mask;
  /* check if addr is within a already selected page */
  for (i=0;i<data->tlb_collision_check_array[index];i++)
  {
    if (tmp==data->tlb_tags[i*(data->tlb_size/data->tlb_sets)+index]) return 0;
  }
  /* check if another page fits into TLB */
  if (data->tlb_collision_check_array[index]<data->tlb_sets)
  {
    data->tlb_tags[data->tlb_collision_check_array[index]*(data->tlb_size/data->tlb_sets)+index]=tmp;
    data->tlb_collision_check_array[index]++;
    return 0;
  }
  
  //printf("TLB_COLLISION\n");
  return -1;
  
}

int already_selected(unsigned long long addr, unsigned long long *addresses, int max)
{
   int i;
   for (i=0;i<max;i++)
   {
      if (addr==addresses[i]) return -1;
   }
   return 0;
}

/**
 * flushes data from the specified cachelevel
 * @param level the cachelevel that should be flushed
 * @param num_flushes number of accesses to each cacheline
 * @param mode FLUSH_RO: use memory read-only
 *             FLUSH_RW: use memory read-write 
 * @param buffer pointer to a memory area, size of the buffer has to be 
 *               has to be larger than 2 x sum of all cachelevels <= level
 * @return 0 if successful, -1 on error
 */
int inline cacheflush(int level,int num_flushes,int mode,void* buffer,cpu_info_t cpuinfo)
{
  unsigned long long stride=cpuinfo.Cacheline_size[level-1]/num_flushes;
  unsigned long long size=0;
  unsigned int i,j,tmp=0x0fa38b09;

  if (level>cpuinfo.Cachelevels) return -1;

  if (!strcmp(cpuinfo.vendor,"AuthenticAMD"))for (i=0;i<level;i++)
  {
     if (cpuinfo.Cache_unified[i]) size+=cpuinfo.U_Cache_Size[i];
     else size+=cpuinfo.D_Cache_Size[i];
  }
  if (!strcmp(cpuinfo.vendor,"GenuineIntel"))
  {
     i=level-1;
     if (cpuinfo.Cache_unified[i]) size=cpuinfo.U_Cache_Size[i];
     else size=cpuinfo.D_Cache_Size[i];
  } 

  size*=12;
  size/=10;

  if (stride<sizeof(unsigned int)) stride=sizeof(unsigned int);
  
  j=num_flushes;
  while(j--)
  {
    for (i=0;i<size;i+=stride)
    {
      //tmp|=*((int*)((unsigned long long)buffer+i));
      *((int*)((unsigned long long)buffer+i))=tmp;
    }
  }
  if (mode==MODE_EXCLUSIVE) 
  {
    clflush(buffer,size,cpuinfo);
    j=num_flushes;
    while(j--)
    {
     for (i=0;i<size;i+=stride)
     {
       tmp|=*((int*)((unsigned long long)buffer+i));
     }
     *((int*)((unsigned long long)buffer+i))=tmp;
    }
  }
  if (mode==MODE_INVALID) 
  {
    clflush(buffer,size,cpuinfo);
  }

  return 0;
}

/*
 * flush all caches that are smaller than the specified memory size, including shared caches
 */
void inline flush_caches(void* buffer,unsigned long long memsize,int settings,int num_flushes,int flush_mode,void* flush_buffer,cpu_info_t *cpuinfo)
{
   int i,j;
   unsigned long long total_cache_size;
   if (!strcmp(cpuinfo->vendor,"AuthenticAMD")) //exclusive caches
   for (i=cpuinfo->Cachelevels;i>0;i--)
   {   
     if (settings&FLUSH(i))
     {
       total_cache_size=0;
       for (j=i;j>0;j--) total_cache_size+=cpuinfo->U_Cache_Size[j-1]+cpuinfo->D_Cache_Size[j-1];
       if(memsize>total_cache_size)
       {
         cacheflush(i,num_flushes,flush_mode,flush_buffer,*(cpuinfo));
         break;
       }
     }
   }   
   else // inclusive caches
   for (i=cpuinfo->Cachelevels;i>0;i--)
   {   
     if ((settings&FLUSH(i))&&(memsize>(cpuinfo->U_Cache_Size[i-1]+cpuinfo->D_Cache_Size[i-1])))
     {
       cacheflush(i,num_flushes,flush_mode,flush_buffer,*(cpuinfo));
       break;
     }
   }
}

/*
 * use a block of memory to ensure it is in the caches afterwards
 */
void inline use_memory(void* buffer,unsigned long long memsize,int mode,int repeat,cpu_info_t cpuinfo,volatile mydata_t *data, threaddata_t *threaddata)
{
   int i,j,tmp=0xd08a721b;
   unsigned long long stride = 128;
   unsigned long long tmp_addr,tmp_offset,mask,max_accesses;
   unsigned long long usable_memory,num_pages,accesses_per_page,usable_page_size;
   struct timeval time;

   /* aligned address */
   unsigned long long aligned_addr;
   
   aligned_addr=(unsigned long long)buffer;
   
   //printf("use_mode: 0x%x\n",mode);fflush(stdout);

   for (i=cpuinfo.Cachelevels;i>0;i--)
   {
     if (cpuinfo.Cacheline_size[i-1]<stride) stride=cpuinfo.Cacheline_size[i-1];
   }
   
   if ((mode==MODE_MODIFIED)||(mode==MODE_EXCLUSIVE)||(mode==MODE_INVALID))
   {  
   #ifdef RANDOMIZE_3
   /* clear the memory */
   memset(buffer,0,memsize);
   
   mask=(data->pagesize-1)^0xffffffffffffffffULL;
   usable_memory=(memsize&mask);
   usable_page_size=data->pagesize;

    if ((data->settings&RESTORE_TLB)&&(data->hugepages==HUGEPAGES_OFF))
    {
      usable_memory=usable_memory/2;
      usable_page_size=(data->pagesize)/2;
      if (usable_memory>data->tlb_size*(data->pagesize/2)) usable_memory=data->tlb_size*(data->pagesize/2);
      reset_tlb_check(data);
    }

    max_accesses=(usable_memory/alignment);
    if (max_accesses<accesses) accesses=max_accesses;
    num_pages=usable_memory/usable_page_size;
    accesses=(accesses/32)*32;
    if (accesses<=num_pages) {num_pages=accesses;usable_memory=num_pages*usable_page_size;/*alignment=usable_page_size;*/}

      gettimeofday( &time, (struct timezone *) 0);
      _random_init(time.tv_sec*time.tv_usec+pthread_self()*iteration*iteration,memsize/data->pagesize);
      //randomly select pages (4KB)
      for (j=0;j<num_pages;j++)
      {
        data->page_address[j]=(((unsigned long long)_random())*data->pagesize);
        while ((tlb_check(aligned_addr+data->page_address[j],data))||(already_selected(aligned_addr+data->page_address[j],data->page_address,j)))
        {
           data->page_address[j]+=data->pagesize;
           if (data->page_address[j]>=(memsize&mask)) data->page_address[j]=0; 
        }
        if (threaddata!=NULL) threaddata->page_address[j]=data->page_address[j]; 
        data->page_address[j]+=aligned_addr;
        if (threaddata!=NULL) threaddata->page_address[j]+=threaddata->aligned_addr;
      }  
  
      //randomly select addresses within the choosen pages
      //bi_random_init(time.tv_usec,usable_memory); 
      gettimeofday( &time, (struct timezone *) 0);
      _random_init(time.tv_sec*time.tv_usec+pthread_self()*iteration*iteration,usable_memory/alignment);
      tmp_addr=aligned_addr;
      for(j=0;j<accesses;j++)
      {
        //tmp_offset=(((unsigned long long)bi_random32())&(((unsigned long long)alignment-1)^0xffffffffffffffffULL));
        tmp_offset=(((unsigned long long)_random())*alignment);

        while (((*((unsigned long long*)(data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size)))!=0)&&(j!=accesses-1))||((data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size))==tmp_addr))
        {
           tmp_offset+=alignment;
           if (tmp_offset>=usable_memory) tmp_offset=0;  
        }    
        //printf("%16x - page %i, offset %i\n",tmp_offset,tmp_offset/(data->pagesize/2),tmp_offset%(data->pagesize/2));fflush(stdout);  
        *((unsigned long long*)(tmp_addr))=data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size);
        //if(threaddata!=NULL) *((unsigned long long*)((tmp_addr-aligned_addr)+threaddata->aligned_addr))=threaddata->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size);

        tmp_addr=data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size);
      }
      #endif
    
     j=repeat;
     while(j--)
     {
       for (i=0;i<memsize-stride;i+=stride)
       {
         //this kernel needs the content of the buffer, so the usage must not be destructive
         //compiler optimisation has to be disabled (-O0), otherwise the compiler could remove the following commands
         tmp=*((int*)((unsigned long long)buffer+i));
         *((int*)((unsigned long long)buffer+i))=tmp;
       }
     }
     //now buffer is invalid in other caches, modified in local cache
   }
   if ((mode==MODE_EXCLUSIVE)||(mode==MODE_SHARED)||(mode==MODE_OWNED)||(mode==MODE_FORWARD))
   {
     if (mode==MODE_EXCLUSIVE) 
     {
      clflush(buffer,memsize,cpuinfo);
      //now buffer is invalid in local cache
     }
     j=repeat;
     while(j--)
     {
      for (i=0;i<memsize-stride;i+=stride)
      {
        tmp|=*((int*)((unsigned long long)buffer+i));
      }
      //result has to be stored somewhere to prevent the compiler from deleting the hole loop
      //if compiler optimisation is disabled, the following command is not needed
      //*((int*)((unsigned long long)buffer+i))=tmp;
     }
     //now buffer is exclusive or shared in local cache
   }


/*   if (mode==MODE_SHARED)
   {
      char* tmp;
      tmp=(char*)malloc(cpuinfo.Total_D_Cache_Size);
      cacheflush(3,4,MODE_MODIFIED,tmp,cpuinfo);
      free(tmp);
      //clflush(buffer,memsize,cpuinfo); 
   }
*/   
   
   if (mode==MODE_INVALID)
   {
     clflush(buffer,memsize,cpuinfo);
     //now buffer is invalid in local cache too
   }
}

/*
 * loop for additional worker threads
 * communicating with master thread using shared variables
 */
void *thread(void *threaddata)
{
  int id= ((threaddata_t *) threaddata)->thread_id;
  volatile mydata_t* global_data = ((threaddata_t *) threaddata)->data; //communication
  threaddata_t* mydata = (threaddata_t*)threaddata;
  char* filename=NULL;

  struct timespec wait_ns;
  int i,j,k,fd;
  double tmp=(double)0;
  unsigned long long tmp2,tmp3,old=THREAD_STOP;
  
  wait_ns.tv_sec=0;
  wait_ns.tv_nsec=100000;

  do
  {
   old=global_data->thread_comm[id];
  }
  while (old!=THREAD_INIT);
  global_data->ack=id;

  cpu_set(((threaddata_t *) threaddata)->cpu_id);

  if(mydata->buffersize)
  {
    if (global_data->hugepages==HUGEPAGES_OFF) mydata->buffer = _mm_malloc( mydata->buffersize,mydata->alignment);
    if (global_data->hugepages==HUGEPAGES_ON)
    {
      char *dir;
      dir=bi_getenv("BENCHIT_KERNEL_HUGEPAGE_DIR",0);
      filename=(char*)malloc((strlen(dir)+20)*sizeof(char));
      sprintf(filename,"%s/thread_data_%i",dir,id);
      mydata->buffer=NULL;
      fd=open(filename,O_CREAT|O_RDWR,0664);
      if (fd == -1)
      {
        fprintf( stderr, "Allocation of buffer failed\n" ); fflush( stderr );
        perror("open");
        exit( 127 );
      } 
      mydata->buffer=(char*) mmap(NULL,mydata->buffersize,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
      close(fd);unlink(filename);
    } 
    //fill buffer
    tmp2=sizeof(unsigned long long);
    for (i=0;i<mydata->buffersize;i+=tmp2)
    {
      *((unsigned long long*)((unsigned long long)mydata->buffer+i))=(unsigned long long)i;
    }
    clflush(mydata->buffer,mydata->buffersize,*(mydata->cpuinfo));
    mydata->aligned_addr=(unsigned long long)(mydata->buffer) + mydata->offset;
  }
  else mydata->aligned_addr=(unsigned long long)(global_data->buffer) + mydata->offset;
  
  while(1)
  {
     //printf("Thread %i: comm= %i\n",id+1,data->thread_comm[id]);
     switch (global_data->thread_comm[id]){
       case THREAD_USE_MEMORY: 
         if (old!=THREAD_USE_MEMORY)
         {
           global_data->ack=id;old=THREAD_USE_MEMORY;
           
           //clear caches
           //if ((mydata->USE_MODE==MODE_EXCLUSIVE)||(mydata->USE_MODE==MODE_MODIFIED)) flush_caches((void*)(mydata->aligned_addr),global_data->cpuinfo->Total_D_Cache_Size,global_data->settings,global_data->NUM_FLUSHES,MODE_INVALID,global_data->cache_flush_area,global_data->cpuinfo);
   
           // use memory
           use_memory((void*)mydata->aligned_addr,mydata->memsize,mydata->USE_MODE,mydata->NUM_USES,*(mydata->cpuinfo),global_data,mydata);

           //flush cachelevels as specified in PARAMETERS
           flush_caches((void*) (mydata->aligned_addr),mydata->memsize,mydata->settings,mydata->NUM_FLUSHES,mydata->FLUSH_MODE,mydata->cache_flush_area,mydata->cpuinfo);
 
           global_data->done=id;
         }
         else 
         {
           tmp=100;while(tmp>0) tmp--;
         }        
         break;
       case THREAD_WAIT: // waiting
          if (old!=THREAD_WAIT) {global_data->ack=id;old=THREAD_WAIT;}
          tmp=100;while(tmp) tmp--;  
          break;
       case THREAD_FLUSH: // flush caches
          if (old!=THREAD_FLUSH)
          {
            global_data->ack=id;
            old=THREAD_FLUSH;
            cacheflush(mydata->cpuinfo->Cachelevels,1,MODE_INVALID,mydata->cache_flush_area,*(mydata->cpuinfo));
            global_data->done=id;
          }
          tmp=100;while(tmp) tmp--; 
          break;
       case THREAD_INIT: // used for parallel initialisation only
          tmp=100;while(tmp) tmp--; 
          break;
       case THREAD_STOP: // exit
       default:
         if (global_data->hugepages==HUGEPAGES_ON)
         {
           if(mydata->buffer!=NULL) munmap((void*)mydata->buffer,mydata->buffersize);
         }
         pthread_exit(NULL);
    }
  }
}

/*
 * assembler implementation of latency measurement
 */
static int asm_work(unsigned long long addr, unsigned long long passes,volatile mydata_t *data) __attribute__((noinline));
static int asm_work(unsigned long long addr, unsigned long long passes,volatile mydata_t *data)
{
   unsigned long long a,b;
   int i;

   if (!passes) return 0;

   #ifdef USE_PAPI
    if (data->num_events) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif

   
   /*
   * Input: addr -> RBX (pointer to the buffer)
   *        passes -> RCX (loop iterations)
   * Output : RAX stop timestamp
   *          RBX start timestamp
   */
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                //the unrolled loop jumping across memory (the memory contains precalculated random target addresses)
                "read_loop: mov (%%rbx), %%rbx;" 
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "mov (%%rbx), %%rbx;"
                "sub $1,%%rcx;"
                "jnz read_loop;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                "mov %%rdx,%%rbx;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                : "=a" (a),"=b" (b)
                : "b"(addr), "c" (passes)
                : "%rdx"
								);
						
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (data->num_events) PAPI_read(data->Eventset,data->values);
    for (i=0;i<data->num_events;i++)
    {
       data->papi_results[i]=(double)data->values[i]/(double)(passes*32);
    }
  #endif
  //printf("end asm\n");fflush(stdout);
	return (unsigned int) ((a-b)-data->cpuinfo->rdtsc_latency)/(passes*32);
}

/*
 * function that does the measurement
 */
void _work( int memsize, int def_alignment, int offset,  int num_accesses, int runs, volatile mydata_t* data, double **results)
{
  int i,j,k,t,tmin;
  unsigned long long tmp,tmp2,tmp3,mask;
  
  unsigned long long usable_memory,num_pages,accesses_per_page,usable_page_size;
	
  unsigned long long tmp_addr,tmp_offset,max_accesses;	

   /* aligned address */
   unsigned long long aligned_addr;

  struct timeval time;
  
  gettimeofday( &time, (struct timezone *) 0);
  
	/* calculate aligned address*/
  aligned_addr = (unsigned long long)(data->buffer)+offset;
  
  accesses=num_accesses;
  alignment=def_alignment;
  //printf("addr        : %018p\n",data->buffer);
  //printf("aligned_addr: 0x%016lx\n",aligned_addr);fflush(stdout);

  /* clear the memory */
  //memset((void*)aligned_addr,0,memsize);  

  mask=(data->pagesize-1)^0xffffffffffffffffULL;
  usable_memory=(memsize&mask);
  usable_page_size=data->pagesize;

  if ((data->settings&RESTORE_TLB)&&(data->hugepages==HUGEPAGES_OFF))
  {
    usable_memory=usable_memory/2;
    usable_page_size=(data->pagesize)/2;
    if (usable_memory>data->tlb_size*(data->pagesize/2)) usable_memory=data->tlb_size*(data->pagesize/2);
    reset_tlb_check(data);
  }

  max_accesses=(usable_memory/alignment);
  if (max_accesses<accesses) accesses=max_accesses;
  num_pages=usable_memory/usable_page_size;
  accesses=(accesses/32)*32;
  if (accesses<=num_pages) {num_pages=accesses;usable_memory=num_pages*usable_page_size;/*alignment=usable_page_size;*/}

  data->page_address=malloc(sizeof(unsigned long long)*num_pages);
  for (t=1;t<data->num_threads;t++) data->threaddata[t].page_address=malloc(sizeof(unsigned long long)*num_pages);

  #ifdef RANDOMIZE_1
      /* clear the memory */
      memset((void*)aligned_addr,0,memsize);

      gettimeofday( &time, (struct timezone *) 0);
      _random_init(time.tv_usec+i*t,memsize/data->pagesize);
      //randomly select pages (4KB)
      for (j=0;j<num_pages;j++)
      {
        data->page_address[j]=(((unsigned long long)_random())*data->pagesize);
        while ((tlb_check(aligned_addr+data->page_address[j],data))||(already_selected(aligned_addr+data->page_address[j],data->page_address,j)))
        {
           data->page_address[j]+=data->pagesize;
           if (data->page_address[j]>=(memsize&mask)) data->page_address[j]=0; 
        }
        for (t=1;t<data->num_threads;t++) data->threaddata[t].page_address[j]=data->page_address[j]; 
        data->page_address[j]+=aligned_addr;
        for (t=1;t<data->num_threads;t++) data->threaddata[t].page_address[j]+=data->threaddata[t].aligned_addr;
      }  
  
      //randomly select addresses within the choosen pages
      //bi_random_init(time.tv_usec,usable_memory);
      gettimeofday( &time, (struct timezone *) 0);
      _random_init(time.tv_usec+i*t,usable_memory/alignment);
      tmp_addr=aligned_addr;
      for(j=0;j<accesses;j++)
      {
        //tmp_offset=(((unsigned long long)bi_random32())&(((unsigned long long)alignment-1)^0xffffffffffffffffULL));
        tmp_offset=(((unsigned long long)_random())*alignment);

        while (((*((unsigned long long*)(data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size)))!=0)&&(j!=accesses-1))||((data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size))==tmp_addr))
        {
           tmp_offset+=alignment;
           if (tmp_offset>=usable_memory) tmp_offset=0;  
        }    
        //printf("%16x - page %i, offset %i\n",tmp_offset,tmp_offset/(data->pagesize/2),tmp_offset%(data->pagesize/2));fflush(stdout);  
        *((unsigned long long*)(tmp_addr))=data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size);
        for (t=1;t<data->num_threads;t++) *((unsigned long long*)((tmp_addr-aligned_addr)+data->threaddata[t].aligned_addr))=data->threaddata[t].page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size);

        tmp_addr=data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size);
      }
  #endif
  //printf("starting measurment %i accesses in %i Bytes of memory\n",accesses,memsize);fflush(stdout);
  for (t=0;t<data->num_threads;t++)
  {
   #ifdef AVERAGE
    tmin=0;
   #else
    tmin=1000000;
   #endif
   
   if (accesses>=32) 
   {

    for (i=0;i<runs;i++)
    {
      iteration=i;
      #ifdef RANDOMIZE_2
      /* clear the memory */
      memset((void*)aligned_addr,0,memsize);

      gettimeofday( &time, (struct timezone *) 0);
      _random_init(time.tv_usec+i*t,memsize/data->pagesize);
      //randomly select pages (4KB)
      for (j=0;j<num_pages;j++)
      {
        data->page_address[j]=(((unsigned long long)_random())*data->pagesize);
        while ((tlb_check(aligned_addr+data->page_address[j],data))||(already_selected(aligned_addr+data->page_address[j],data->page_address,j)))
        {
           data->page_address[j]+=data->pagesize;
           if (data->page_address[j]>=(memsize&mask)) data->page_address[j]=0; 
        }
        if (t) data->threaddata[t].page_address[j]=data->page_address[j]; 
        data->page_address[j]+=aligned_addr;
        if (t) data->threaddata[t].page_address[j]+=data->threaddata[t].aligned_addr;
      }  
  
      //randomly select addresses within the choosen pages
      //bi_random_init(time.tv_usec,usable_memory);
      gettimeofday( &time, (struct timezone *) 0);
      _random_init(time.tv_usec+i*t,usable_memory/alignment);
      tmp_addr=aligned_addr;
      for(j=0;j<accesses;j++)
      {
        //tmp_offset=(((unsigned long long)bi_random32())&(((unsigned long long)alignment-1)^0xffffffffffffffffULL));
        tmp_offset=(((unsigned long long)_random())*alignment);

        while (((*((unsigned long long*)(data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size)))!=0)&&(j!=accesses-1))||((data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size))==tmp_addr))
        {
           tmp_offset+=alignment;
           if (tmp_offset>=usable_memory) tmp_offset=0;  
        }    
        //printf("%16x - page %i, offset %i\n",tmp_offset,tmp_offset/(data->pagesize/2),tmp_offset%(data->pagesize/2));fflush(stdout);  
        *((unsigned long long*)(tmp_addr))=data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size);
        if(t) *((unsigned long long*)((tmp_addr-aligned_addr)+data->threaddata[t].aligned_addr))=data->threaddata[t].page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size);

        tmp_addr=data->page_address[tmp_offset/usable_page_size]+(tmp_offset%usable_page_size);
      }
      #endif
 
    //clear caches
    //TODO reenable if tlb warmup is disabled
    //if (t) flush_caches((void*)(data->threaddata[t].aligned_addr),data->cpuinfo->Total_D_Cache_Size,data->settings,data->NUM_FLUSHES,MODE_INVALID,data->cache_flush_area,data->cpuinfo);
    //else flush_caches((void*)aligned_addr,data->cpuinfo->Total_D_Cache_Size,data->settings,data->NUM_FLUSHES,MODE_INVALID,data->cache_flush_area,data->cpuinfo);

     //access whole buffer to warm up tlb
     if (t) use_memory((void*)(data->threaddata[t].aligned_addr),memsize,MODE_INVALID,data->NUM_USES,*(data->cpuinfo),data,NULL);

     if ((data->USE_MODE==MODE_FORWARD))
      {
        //tell another thread to use memory
        unsigned long long tmp;
        tmp=data->threaddata[data->SHARE_CPU].aligned_addr;
        if (t) data->threaddata[data->SHARE_CPU].aligned_addr=data->threaddata[t].aligned_addr;
        else data->threaddata[data->SHARE_CPU].aligned_addr=aligned_addr;
        data->threaddata[data->SHARE_CPU].memsize=memsize;
        data->threaddata[data->SHARE_CPU].accesses=accesses;
        data->threaddata[data->SHARE_CPU].USE_MODE=MODE_EXCLUSIVE;
        __asm__ __volatile__("mfence;"::);
        data->thread_comm[data->SHARE_CPU]=THREAD_USE_MEMORY;
        while (!data->ack);
        data->ack=0;
        data->thread_comm[data->SHARE_CPU]=THREAD_WAIT;    
        //wait for other thread using the memory
        while (!data->ack); //printf("wait for ack 3\n");
        data->ack=0;
        while (!data->done); //printf("wait for done 3\n");
        data->done=0;
        data->threaddata[data->SHARE_CPU].aligned_addr=tmp;
      }
      
      if (!t) 
      {      
        //access whole buffer to warm up cache and tlb
        if ((data->USE_MODE==MODE_SHARED)) use_memory((void*)aligned_addr,memsize,MODE_EXCLUSIVE,data->NUM_USES,*(data->cpuinfo),data,NULL);
        else if ((data->USE_MODE==MODE_OWNED)) use_memory((void*)aligned_addr,memsize,MODE_MODIFIED,data->NUM_USES,*(data->cpuinfo),data,NULL);
        else use_memory((void*)aligned_addr,memsize,data->USE_MODE,data->NUM_USES,*(data->cpuinfo),data,NULL);
      }

      if (t)
      {
        //tell other thread to use memory
        data->threaddata[t].memsize=memsize;
        data->threaddata[t].accesses=accesses;
        if ((data->USE_MODE==MODE_SHARED)) data->threaddata[t].USE_MODE=MODE_EXCLUSIVE;
        if ((data->USE_MODE==MODE_OWNED)) data->threaddata[t].USE_MODE=MODE_MODIFIED;
        __asm__ __volatile__("mfence;"::);
        data->thread_comm[t]=THREAD_USE_MEMORY;
        while (!data->ack);
        data->ack=0;
        data->thread_comm[t]=THREAD_WAIT;    
        //wait for other thread using the memory
        while (!data->ack); //printf("wait for ack 2\n");
        data->ack=0;
        while (!data->done);//printf("wait for done 2\n");
        data->done=0;     
        
      }
     
      //flush cachelevels as specified in PARAMETERS
      //flush_caches((void*) data->threaddata[t].aligned_addr,memsize,data->settings,data->NUM_FLUSHES,data->FLUSH_MODE,data->cache_flush_area,data->cpuinfo);
          
      if ((data->USE_MODE==MODE_SHARED)||(data->USE_MODE==MODE_OWNED))
      {
        //tell another thread to use memory
        unsigned long long tmp;
        tmp=data->threaddata[data->SHARE_CPU].aligned_addr;
        if (t) data->threaddata[data->SHARE_CPU].aligned_addr=data->threaddata[t].aligned_addr;
        else data->threaddata[data->SHARE_CPU].aligned_addr=aligned_addr;
        data->threaddata[data->SHARE_CPU].memsize=memsize;
        data->threaddata[data->SHARE_CPU].accesses=accesses;
        data->threaddata[data->SHARE_CPU].USE_MODE=data->USE_MODE;
        __asm__ __volatile__("mfence;"::);
        data->thread_comm[data->SHARE_CPU]=THREAD_USE_MEMORY;
        while (!data->ack);
        data->ack=0;
        data->thread_comm[data->SHARE_CPU]=THREAD_WAIT;    
        //wait for other thread using the memory
        while (!data->ack); //printf("wait for ack 3\n");
        data->ack=0;
        while (!data->done); //printf("wait for done 3\n");
        data->done=0;
        data->threaddata[data->SHARE_CPU].aligned_addr=tmp;
      }

      //flush cachelevels as specified in PARAMETERS
      flush_caches((void*) data->threaddata[t].aligned_addr,memsize,data->settings,data->NUM_FLUSHES,data->FLUSH_MODE,data->cache_flush_area,data->cpuinfo);

      //TODO check if this is still working after changes to usable memory size
      //restore TLB if enabled (that was destroied by flushing the cache)
      if ((data->settings&RESTORE_TLB)&&(data->hugepages==HUGEPAGES_OFF))
      {
       //printf("restore TLB\n");fflush(stdout);
        tmp2=data->pagesize/2+data->pagesize/8;
        for (j=0;j<num_pages;j++)
        {
          for (k=tmp2;k<tmp2+data->pagesize/4;k+=alignment)
          {
            if (!t)
            {
              tmp=*((unsigned long long*)(data->page_address[j]+k));
              *((unsigned long long*)(data->page_address[j]+k))=tmp;
              clflush((void*)data->page_address,num_pages*sizeof(unsigned long long),*(data->cpuinfo));  
            }
           else
            {
              tmp=*((unsigned long long*)(data->threaddata[t].page_address[j]+k));
              *((unsigned long long*)(data->threaddata[t].page_address[j]+k))=tmp;
              clflush((void*)data->threaddata[t].page_address,num_pages*sizeof(unsigned long long),*(data->cpuinfo));  
            }
          }
        }
      }
                  

      /* call ASM implementation */
      //printf("call asm impl.\n");
      if (!t) tmp=asm_work(aligned_addr,accesses/32,data);
      else tmp=asm_work(data->threaddata[t].aligned_addr,accesses/32,data); 
      //printf("left asm impl.\n");fflush(stdout);
      if (tmp!=-1)
      {
       #ifdef AVERAGE
         tmin+=tmp;
       #else
         if (tmp<tmin) tmin=tmp;
       #endif       
      }
    }
    #ifdef AVERAGE
      tmin/=runs;
    #endif
   }
   else tmin=0;
  
   if (tmin) (*results)[t]=(double)tmin;
   else (*results)[t]=INVALID_MEASUREMENT;
  }
  if (data->page_address) free(data->page_address); 
  for (t=1;t<data->num_threads;t++) if (data->threaddata[t].page_address) free(data->threaddata[t].page_address);
}
