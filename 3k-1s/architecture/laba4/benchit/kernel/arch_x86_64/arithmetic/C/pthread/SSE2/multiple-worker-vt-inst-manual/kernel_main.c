/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures aggregate read bandwidth of multiple parallel threads.
 *******************************************************************/

#define _GNU_SOURCE
#include <sched.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "interface.h"
#include <assert.h>
#include <signal.h>
#include <errno.h>

/*  Header for local functions
 */
#include "work.h"

#ifdef USE_PAPI
#include <papi.h>
#endif

#ifdef USE_VTRACE
#include "vt_user.h"
#endif

/** These variables will help us to keep the overview over the arrays
  * we access for our functions/data.
  */
int n_of_works;
int n_of_sure_funcs_per_work;

/*
 * variables to store settings from PARAMETERS file
 * set by evaluate_environment function
 */
int ALLOCATION, HUGEPAGES, OFFSET=0, ALIGNMENT=0, BUFFERSIZE;
int NUM_USES=0,USE_MODE,NUM_THREADS=0;
int THREAD_OFFSET=0,USE_DIRECTION;
int BURST_LENGTH,FUNCTION,TIMEOUT=0;

long long INT_INIT[2]={1,-1};
double FP_INIT[2]={2.0,0.5};

unsigned long long ACCESSES;

char *error_msg=NULL;

/*
 * CPU bindings of threads, derived from CPU_LIST in PARAMETERS file
 */
cpu_set_t cpuset;
unsigned long long *cpu_bind;

/*
 * filename and filedescriptor for hugetlbfs
 */
char* filename;
int fd;

/*
 * data structure for hardware detection
 */
cpu_info_t *cpuinfo=NULL;

/*
 * needed for cacheflush function, estimated by hardware detection
 */
int CACHEFLUSHSIZE=0,L1_SIZE=-1,L2_SIZE=-1,L3_SIZE=-1,CACHELINE=0,CACHELEVELS=0;

/*
 * needed to calculate duration from clock cycles
 */
unsigned long long FREQUENCY=0;

/*
 * temporary strings
 */
char buf[256];

/*
 * needed to parse list of problemsizes
 */
int MAX=0;
bi_list_t * problemlist;
unsigned long long problemlistsize;
double *problemarray;

mydata_t* mdp;

/* variables for the PAPI counters*/
#ifdef USE_PAPI
char **papi_names;
int *papi_codes;
int papi_num_counters;
int EventSet;
#endif

/*
 * watchdog timer
 */
pthread_t watchdog;
typedef struct watchdog_args
{
 pid_t pid;
 int timeout;
} watchdog_arg_t;
watchdog_arg_t watchdog_arg;

/* stops watchdog thread if benchmark finishes before timeout*/
static void sigusr1_handler (int signum) {
 pthread_exit(0);
}

/* stops benchmark if timeout is reached*/
static void *watchdog_timer(void *arg)
{
  sigset_t  signal_mask; 
  
  sigemptyset (&signal_mask);
  sigaddset (&signal_mask, SIGINT);
  sigaddset (&signal_mask, SIGTERM);
  pthread_sigmask (SIG_BLOCK, &signal_mask, NULL);
  
  signal(SIGUSR1,sigusr1_handler);
  
  if (((watchdog_arg_t*)arg)->timeout>0)
  {
     sleep(((watchdog_arg_t*)arg)->timeout);
     kill(((watchdog_arg_t*)arg)->pid,SIGTERM);
  }
  pthread_exit(0);
}

void evaluate_environment( bi_info * info );

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with informations about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zero's
 */
void bi_getinfo( bi_info * infostruct )
{
   int i = 0, j = 0; /* loop var for n_of_works */
   char buff[512];
   char tmp[256];
   (void) memset ( infostruct, 0, sizeof( bi_info ) );
   /* get environment variables for the kernel */
   evaluate_environment(infostruct);
   infostruct->codesequence = bi_strdup( "mov mem -> xmm" );
   infostruct->xaxistext = bi_strdup( "data set size [Byte]" );
   infostruct->base_xaxis=10.0;
   infostruct->num_measurements=problemlistsize;
   sprintf(buff,"multi threaded memory bandwidth");
   infostruct->kerneldescription = bi_strdup( buff );
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = NUM_THREADS;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 1;
   
   /* MB/s */
   n_of_works = 1;
   #ifdef USE_PAPI
    n_of_works+=papi_num_counters;
   #endif

   n_of_sure_funcs_per_work = 1;
   
   infostruct->numfunctions = n_of_works * n_of_sure_funcs_per_work;

   /* allocating memory for y axis texts and properties */
   infostruct->yaxistexts = _mm_malloc( infostruct->numfunctions * sizeof( char* ),ALIGNMENT );
   if ( infostruct->yaxistexts == 0 )
   {
     fprintf( stderr, "Allocation of yaxistexts failed.\n" ); fflush( stderr );
     exit( 127 );
   }
   infostruct->selected_result = _mm_malloc( infostruct->numfunctions * sizeof( int ) ,ALIGNMENT);
   if ( infostruct->selected_result == 0 )
   {
     fprintf( stderr, "Allocation of outlier direction failed.\n" ); fflush( stderr );
     exit( 127 );
   }
   *infostruct->selected_result=-1; //average
   infostruct->legendtexts = _mm_malloc( infostruct->numfunctions * sizeof( char* ) ,ALIGNMENT);
   if ( infostruct->legendtexts == 0 )
   {
     fprintf( stderr, "Allocation of legendtexts failed.\n" ); fflush( stderr );
     exit( 127 );
   }
   infostruct->base_yaxis = _mm_malloc( infostruct->numfunctions * sizeof( double ) ,ALIGNMENT);
   if ( infostruct->base_yaxis == 0 )
   {
     fprintf( stderr, "Allocation of base yaxis failed.\n" ); fflush( stderr );
     exit( 127 );
   }
   /* setting up y axis texts and properties */
   for ( j = 0; j < n_of_works; j++ )
   {
      int k,index;
      
      sprintf(buf,"bandwidth %s ",bi_getenv( "BENCHIT_KERNEL_CPU_LIST", 0 ));
                 
      k=0;
      index= k + n_of_sure_funcs_per_work * j;
      infostruct->yaxistexts[index] = bi_strdup( "bandwidth [GB/s]" );
      infostruct->selected_result[index] = 0;
      infostruct->base_yaxis[index] = 0;
      switch ( j )
       {
          case 0:
            infostruct->legendtexts[index] = bi_strdup( buf );
            break;
          default: // papi
            #ifdef USE_PAPI
            sprintf(buff,"%s",papi_names[j-1]);
            infostruct->legendtexts[index] = bi_strdup( buff );
            infostruct->selected_result[index] = 0;
            infostruct->yaxistexts[index] = bi_strdup( "counter value/ memory accesses" );
            #endif
            break;
        } 
   }
   if ( DEBUGLEVEL > 3 )
   {
      /* the next for loop: */
      /* this is for your information only and can be ereased if the kernel works fine */
      for ( i = 0; i < infostruct->numfunctions; i++ )
      {
         printf( "yaxis[%2d]=%s\t\t\tlegend[%2d]=%s\n",
            i, infostruct->yaxistexts[i], i, infostruct->legendtexts[i] );
      }
   }
}

/** Implementation of the bi_init of the BenchIT interface.
 *  Here you have the chance to allocate the memory you need.
 *  It is also possible to allocate the memory at the beginning
 *  of every single measurment and to free the memory thereafter.
 *  But making usage always of the same memory is faster.
 *  HAVE A LOOK INTO THE HOWTO !
 */
void* bi_init( int problemsizemax )
{

   int retval,t,i,j;
   long long tmp;

   #ifdef USE_VTRACE
   VT_USER_START("INIT");
   #endif
/*
   printf("\n");   
   printf("sizeof mydata_t:           %i\n",sizeof(mydata_t));
   printf("sizeof threaddata_t:       %i\n",sizeof(threaddata_t));
   printf("sizeof cpu_info_t:         %i\n",sizeof(cpu_info_t));
*/

   cpu_set(cpu_bind[0]);
   mdp->cpuinfo=cpuinfo;

   BUFFERSIZE=sizeof(char)*(MAX+ALIGNMENT+OFFSET+THREAD_OFFSET*NUM_THREADS+2*sizeof(unsigned long long));
   if (HUGEPAGES==HUGEPAGES_ON) BUFFERSIZE=(BUFFERSIZE+(2*1024*1024))&0xffe00000;
   
   assert(BUFFERSIZE>0);
  
   if (ALLOCATION==ALLOC_GLOBAL)
   {
     if (HUGEPAGES==HUGEPAGES_OFF) mdp->buffer = _mm_malloc( BUFFERSIZE,ALIGNMENT );
     if (HUGEPAGES==HUGEPAGES_ON)
     {
        char *dir;
        dir=bi_getenv("BENCHIT_KERNEL_HUGEPAGE_DIR",0);
        filename=(char*)malloc((strlen(dir)+20)*sizeof(char));
        sprintf(filename,"%s/thread_data_0",dir);
        mdp->buffer=NULL;
        fd=open(filename,O_CREAT|O_RDWR,0664);
        if (fd==-1)
        {
          fprintf( stderr, "Error: could not create file in hugetlbfs\n" ); fflush( stderr );
          perror("open");
          exit( 127 );
        } 
        mdp->buffer=(char*) mmap(NULL,BUFFERSIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        close(fd);unlink(filename);
     } 
     if ((mdp->buffer == 0)||(mdp->buffer == (void*) -1ULL))
     {
       fprintf( stderr, "Error: Allocation of buffer failed\n" ); fflush( stderr );
       if (HUGEPAGES==HUGEPAGES_ON) perror("mmap");
       exit( 127 );
     }
   }

   mdp->settings=0;
 
   /* overwrite hw_detection resulte if specified in PARAMETERS file*/
   if (FREQUENCY)
   {
      //printf("overwriting CPU-Clockrate\n");
      mdp->cpuinfo->clockrate=FREQUENCY;
   }
   else if (mdp->cpuinfo->clockrate==0)
   {
      fprintf( stderr, "Error: CPU-Clockrate could not be estimated\n" );
      exit( 1 );
   }
  
   /* overwrite hw_detection resulte if specified in PARAMETERS file*/
   if(L1_SIZE>=0)
   {
      //printf("overwriting L1-Size\n");
      mdp->cpuinfo->Cacheflushsize-=mdp->cpuinfo->U_Cache_Size[0];
      mdp->cpuinfo->Cacheflushsize-=mdp->cpuinfo->D_Cache_Size[0];
      mdp->cpuinfo->Cacheflushsize+=L1_SIZE;
      mdp->cpuinfo->Cache_unified[0]=0;
      mdp->cpuinfo->Cache_shared[0]=0;
      mdp->cpuinfo->U_Cache_Size[0]=0;
      mdp->cpuinfo->I_Cache_Size[0]=L1_SIZE;
      mdp->cpuinfo->D_Cache_Size[0]=L1_SIZE;
      CACHELEVELS=1;
   }
  
   /* overwrite hw_detection resulte if specified in PARAMETERS file*/
   if(L2_SIZE>=0)
   {
      //printf("overwriting L2-Size\n");
      mdp->cpuinfo->Cacheflushsize-=mdp->cpuinfo->U_Cache_Size[1];
      mdp->cpuinfo->Cacheflushsize-=mdp->cpuinfo->D_Cache_Size[1];
      mdp->cpuinfo->Cacheflushsize+=L2_SIZE;
      mdp->cpuinfo->Cache_unified[1]=0;
      mdp->cpuinfo->Cache_shared[1]=0;
      mdp->cpuinfo->U_Cache_Size[1]=0;
      mdp->cpuinfo->I_Cache_Size[1]=L2_SIZE;
      mdp->cpuinfo->D_Cache_Size[1]=L2_SIZE;
      CACHELEVELS=2;
   }
  
   /* overwrite hw_detection resulte if specified in PARAMETERS file*/
   if(L3_SIZE>=0)
   {
      //printf("overwriting L3-Size\n");
      mdp->cpuinfo->Cacheflushsize-=mdp->cpuinfo->U_Cache_Size[2];
      mdp->cpuinfo->Cacheflushsize-=mdp->cpuinfo->D_Cache_Size[2];
      mdp->cpuinfo->Cacheflushsize+=L3_SIZE;
      mdp->cpuinfo->Cache_unified[2]=0;
      mdp->cpuinfo->Cache_shared[2]=0;
      mdp->cpuinfo->U_Cache_Size[2]=0;
      mdp->cpuinfo->I_Cache_Size[2]=L3_SIZE;
      mdp->cpuinfo->D_Cache_Size[2]=L3_SIZE;
      CACHELEVELS=3;
   }
   
   /* overwrite hw_detection resulte if specified in PARAMETERS file*/
   if (CACHELINE)
   {
      //printf("overwriting Cacheline-length\n");
      mdp->cpuinfo->Cacheline_size[0]=CACHELINE;
      mdp->cpuinfo->Cacheline_size[1]=CACHELINE;
      mdp->cpuinfo->Cacheline_size[2]=CACHELINE;
   }

   printf("\n");
   
   mdp->USE_MODE=USE_MODE;
   mdp->USE_DIRECTION=USE_DIRECTION;
   mdp->INT_INIT=INT_INIT;
   mdp->FP_INIT=FP_INIT;
   
   mdp->hugepages=HUGEPAGES;
   
   if ((NUM_THREADS>mdp->cpuinfo->num_cores)||(NUM_THREADS==0)) NUM_THREADS=mdp->cpuinfo->num_cores;
   
   mdp->num_threads=NUM_THREADS;
   mdp->threads=_mm_malloc(NUM_THREADS*sizeof(pthread_t),ALIGNMENT);
   mdp->thread_comm=_mm_malloc(NUM_THREADS*sizeof(int),ALIGNMENT);
   if ((mdp->threads==NULL)||(mdp->thread_comm==NULL))
   {
     fprintf( stderr, "Error: Allocation of structure mydata_t failed\n" ); fflush( stderr );
     exit( 127 );
   }
   

   if (CACHELEVELS>mdp->cpuinfo->Cachelevels)
   {
      //printf("overwriting Cachelevels\n");
      mdp->cpuinfo->Cachelevels=CACHELEVELS;
   }
   fflush(stdout);

  mdp->threaddata = _mm_malloc(mdp->num_threads*sizeof(threaddata_t),ALIGNMENT);
  
  #ifdef USE_PAPI
   mdp->Eventset=EventSet;
   mdp->num_events=papi_num_counters;
   if (papi_num_counters)
   { 
    mdp->values=(long long*)malloc(papi_num_counters*sizeof(long long));
    mdp->papi_results=(double*)malloc(papi_num_counters*sizeof(double));
   }
   else 
   {
     mdp->values=NULL;
     mdp->papi_results=NULL;
   }
  #else
   mdp->Eventset=0;
   mdp->num_events=0;
   mdp->values=NULL;
  #endif

  if (generic_num_packages()!=-1) mdp->threads_per_package=calloc(generic_num_packages(),sizeof(int));
  else mdp->threads_per_package=calloc(1,sizeof(int));

  for (i=0;i<NUM_THREADS;i++)
  {
    if (get_pkg(cpu_bind[i])!=-1) mdp->threaddata[i].package=get_pkg(cpu_bind[i]);
    else mdp->threaddata[i].package=0;
    mdp->threads_per_package[mdp->threaddata[i].package]++;
    #ifdef UNCORE
    if (mdp->threads_per_package[mdp->threaddata[i].package]==1)
    {
      mdp->threaddata[i].monitor_uncore=1;
      printf("  Thread %i monitors uncore event of package %i\n",i,get_pkg(cpu_bind[i]));
      memset(&(mdp->threaddata[i].ctx),0, sizeof(mdp->threaddata[i].ctx));
      memset(&(mdp->threaddata[i].load_arg), 0, sizeof(mdp->threaddata[i].load_arg));
      mdp->threaddata[i].pc=(pfarg_pmc_t*)calloc(mdp->pfmon_num_events,sizeof(pfarg_pmc_t));
      mdp->threaddata[i].pd=(pfarg_pmd_t*)calloc(mdp->pfmon_num_events,sizeof(pfarg_pmd_t));    
    }
    else mdp->threaddata[i].monitor_uncore=0;
    #endif
  }
 
  #ifdef UNCORE
  if (mdp->threaddata[0].monitor_uncore)
  {
     mdp->threaddata[0].ctx.ctx_flags=PFM_FL_SYSTEM_WIDE;
     mdp->threaddata[0].fd = pfm_create_context(&mdp->threaddata[0].ctx, NULL, NULL, 0);
     if (mdp->threaddata[0].fd == -1) {
        printf("pfm_create_context failed\n");
        exit(-1);
     }
     if (pfm_dispatch_events(&mdp->inp, &mdp->mod_inp, &mdp->outp, NULL) != PFMLIB_SUCCESS) {
        printf("cannot dispatch events\n");
        exit(-1);
     }
     
     for(i=0; i < mdp->outp.pfp_pmc_count; i++) {
        mdp->threaddata[0].pc[i].reg_num   = mdp->outp.pfp_pmcs[i].reg_num;
        mdp->threaddata[0].pc[i].reg_value = mdp->outp.pfp_pmcs[i].reg_value;
     }
     for(i=0; i < mdp->outp.pfp_pmd_count; i++) {
        mdp->threaddata[0].pd[i].reg_num   = mdp->outp.pfp_pmds[i].reg_num;
        mdp->threaddata[0].pd[i].reg_value = 0;
     }   

     if (pfm_write_pmcs(mdp->threaddata[0].fd, mdp->threaddata[0].pc, mdp->outp.pfp_pmc_count) == -1) {
        printf("pfm_write_pmcs failed\n");
        exit(1); 
     }

     if (pfm_write_pmds(mdp->threaddata[0].fd, mdp->threaddata[0].pd, mdp->outp.pfp_pmd_count) == -1) {
        printf("pfm_write_pmds failed\n");
        exit(1);
     }

     mdp->threaddata[0].load_arg.load_pid = cpu_bind[0];
     if (pfm_load_context(mdp->threaddata[0].fd, &(mdp->threaddata[0].load_arg)) == -1) {
        printf("pfm_load_context failed\n");
        perror("");fflush(stdout);fflush(stderr);
        exit(1);
     }
   }
   #endif
  
  // create threads
  for (t=0;t<mdp->num_threads;t++)
  {
    mdp->threaddata[t].cpuinfo=(cpu_info_t*)_mm_malloc( sizeof( cpu_info_t ),ALIGNMENT);
    if ( mdp->cpuinfo == 0 )
    {
      fprintf( stderr, "Error: Allocation of structure mydata_t failed\n" ); fflush( stderr );
      exit( 127 );
    }
    #ifdef USE_PAPI
    if (papi_num_counters)
    { 
     mdp->threaddata[t].values=(long long*)malloc(papi_num_counters*sizeof(long long));
     mdp->threaddata[t].papi_results=(double*)malloc(papi_num_counters*sizeof(double));
    }
    else 
    {
      mdp->threaddata[t].values=NULL;
      mdp->threaddata[t].papi_results=NULL;
    }
    #endif
    memcpy(mdp->threaddata[t].cpuinfo,mdp->cpuinfo,sizeof(cpu_info_t));
    mdp->ack=0;
    mdp->threaddata[t].thread_id=t;
    mdp->threaddata[t].cpu_id=cpu_bind[t];
    mdp->threaddata[t].data=mdp;
    mdp->thread_comm[t]=THREAD_INIT;
    mdp->threaddata[t].settings=mdp->settings;
    if (mdp->cache_flush_area==NULL) mdp->threaddata[t].cache_flush_area=NULL;
    else 
    {
     mdp->threaddata[t].cache_flush_area=(char*)_mm_malloc(mdp->cpuinfo->Cacheflushsize,ALIGNMENT);
     if (mdp->threaddata[t].cache_flush_area == NULL)
     {
        fprintf( stderr, "Error: Allocation of structure mydata_t failed\n" ); fflush( stderr );
        exit( 127 );
     }
     //fill cacheflush-area
     tmp=sizeof(unsigned long long);
     for (i=0;i<mdp->cpuinfo->Cacheflushsize;i+=tmp)
     {
        *((unsigned long long*)((unsigned long long)mdp->threaddata[t].cache_flush_area+i))=(unsigned long long)i;
     }
     clflush(mdp->threaddata[t].cache_flush_area,mdp->cpuinfo->Cacheflushsize,*(mdp->cpuinfo));
    }
    mdp->threaddata[t].NUM_USES=mdp->NUM_USES;
    mdp->threaddata[t].USE_MODE=mdp->USE_MODE;
    if (ALLOCATION==ALLOC_LOCAL) mdp->threaddata[t].buffersize=BUFFERSIZE;
    if (ALLOCATION==ALLOC_GLOBAL) mdp->threaddata[t].buffersize=0;
    mdp->threaddata[t].alignment=ALIGNMENT;
    mdp->threaddata[t].offset=OFFSET;
    mdp->threaddata[t].thread_offset=THREAD_OFFSET;
    mdp->threaddata[t].USE_DIRECTION=USE_DIRECTION;
    mdp->threaddata[t].FUNCTION=FUNCTION;
    mdp->threaddata[t].BURST_LENGTH=BURST_LENGTH;
    mdp->threaddata[t].Eventset=mdp->Eventset;    
    mdp->threaddata[t].num_events=mdp->num_events;
    mdp->threaddata[t].length=ACCESSES;
    
    if (t)
    {
     pthread_create(&(mdp->threads[t]),NULL,thread,(void*)(&(mdp->threaddata[t])));
     while (!mdp->ack);
    }
  }
  mdp->ack=0;mdp->done=0;

  cpu_set(cpu_bind[0]);
 
  if (ALLOCATION==ALLOC_LOCAL)
  {
   if (HUGEPAGES==HUGEPAGES_OFF) mdp->buffer = _mm_malloc( BUFFERSIZE,ALIGNMENT );
   if (HUGEPAGES==HUGEPAGES_ON)
   {
      char *dir;
      dir=bi_getenv("BENCHIT_KERNEL_HUGEPAGE_DIR",0);
      filename=(char*)malloc((strlen(dir)+20)*sizeof(char));
      sprintf(filename,"%s/thread_data_0",dir);
      mdp->buffer=NULL;
      fd=open(filename,O_CREAT|O_RDWR,0664);
      if (fd == -1)
      {
        fprintf( stderr, "Error: could not create file in hugetlbfs\n" ); fflush( stderr );
        perror("open");
        exit( 127 );
      } 
      mdp->buffer=(char*) mmap(NULL,BUFFERSIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
      close(fd);unlink(filename);
   } 
   if ((mdp->buffer == 0)||(mdp->buffer == (void*) -1ULL))
   {
      fprintf( stderr, "Error: Allocation of buffer failed\n" ); fflush( stderr );
      if (HUGEPAGES==HUGEPAGES_ON) perror("mmap");
      exit( 127 );
   }
  }

  //fill buffer
  switch (FUNCTION)
  {
 
   /*case USE_MUL_PI:
   case USE_ADD_PI: 

   tmp=8*BURST_LENGTH*sizeof(int);
   for (i=0;i<=BUFFERSIZE-tmp;i+=tmp)
   {
      for(j=0;j<4*BURST_LENGTH;j++)
        *((int*)((unsigned long long)mdp->buffer+i+j*sizeof(int)))=(int)INT_INIT[0];
      for(j=4*BURST_LENGTH;j<4*BURST_LENGTH;j++)
        *((int*)((unsigned long long)mdp->buffer+i+j*sizeof(int)))=(int)INT_INIT[1];
   }
   break;*/
  
   case USE_LOAD_PI:
   case USE_STORE:
   case USE_STORE_NT:
   case USE_COPY:
   case USE_COPY_NT:
   case USE_SCALE_INT:
   case USE_MUL_PI:
   case USE_ADD_PI:
   case USE_MUL_SI:
   case USE_ADD_SI:
   case USE_AND_PI:
   
   tmp=4*BURST_LENGTH*sizeof(long long);
   for (i=0;i<=BUFFERSIZE-tmp;i+=tmp)
   {
      for(j=0;j<2*BURST_LENGTH;j++)
        *((long long*)((unsigned long long)mdp->buffer+i+j*sizeof(long long)))=INT_INIT[0];
      for(j=2*BURST_LENGTH;j<4*BURST_LENGTH;j++)
        *((long long*)((unsigned long long)mdp->buffer+i+j*sizeof(long long)))=INT_INIT[1];
   }
   break;
   
   case USE_LOAD_PD:
   case USE_MUL_PD:
   case USE_ADD_PD:
   case USE_MUL_SD:
   case USE_ADD_SD:
   case USE_DIV_PD:
   case USE_DIV_SD:
   case USE_SQRT_PD:
   case USE_SQRT_SD:
   case USE_AND_PD:
   case USE_MUL_ADD_PD:
   case USE_MUL_PLUS_ADD_PD:
   
   /* avoid FP overflows:
      create x, -1/x, x, -1/x, -x, 1/x, -x, 1/x pattern to guarantee stable register values for add, mul, and mul+add 
      (i.e sum = 0, product = 1, and sum of all partial products = 0) */   
   tmp=16*BURST_LENGTH*sizeof(double);
   for (i=0;i<=BUFFERSIZE-tmp;i+=tmp)
   {   
      for(j=0;j<2*BURST_LENGTH;j++)
        *((double*)((unsigned long long)mdp->buffer+i+j*sizeof(double)))=FP_INIT[0];
      for(j=2*BURST_LENGTH;j<4*BURST_LENGTH;j++)
        *((double*)((unsigned long long)mdp->buffer+i+j*sizeof(double)))=-1.0*FP_INIT[1];
      for(j=4*BURST_LENGTH;j<6*BURST_LENGTH;j++)
        *((double*)((unsigned long long)mdp->buffer+i+j*sizeof(double)))=FP_INIT[0];
      for(j=6*BURST_LENGTH;j<8*BURST_LENGTH;j++)
        *((double*)((unsigned long long)mdp->buffer+i+j*sizeof(double)))=-1.0*FP_INIT[1];
      for(j=8*BURST_LENGTH;j<10*BURST_LENGTH;j++)
        *((double*)((unsigned long long)mdp->buffer+i+j*sizeof(double)))=-1.0*FP_INIT[0];
      for(j=10*BURST_LENGTH;j<12*BURST_LENGTH;j++)
        *((double*)((unsigned long long)mdp->buffer+i+j*sizeof(double)))=FP_INIT[1];
      for(j=12*BURST_LENGTH;j<14*BURST_LENGTH;j++)
        *((double*)((unsigned long long)mdp->buffer+i+j*sizeof(double)))=-1.0*FP_INIT[0];
      for(j=14*BURST_LENGTH;j<16*BURST_LENGTH;j++)
        *((double*)((unsigned long long)mdp->buffer+i+j*sizeof(double)))=FP_INIT[1];
    }
    //printf("%f: %f - %f - %f - %f - %f - %f - %f - %f\n",FP_INIT[0],((double*)(mdp->buffer))[0],((double*)(mdp->buffer))[1],((double*)(mdp->buffer))[2],((double*)(mdp->buffer))[3],((double*)(mdp->buffer))[4],((double*)(mdp->buffer))[5],((double*)(mdp->buffer))[6],((double*)(mdp->buffer))[7]);
    //printf("%f: %f - %f - %f - %f - %f - %f - %f - %f\n",FP_INIT[1],((double*)(mdp->buffer))[8],((double*)(mdp->buffer))[9],((double*)(mdp->buffer))[10],((double*)(mdp->buffer))[11],((double*)(mdp->buffer))[12],((double*)(mdp->buffer))[13],((double*)(mdp->buffer))[14],((double*)(mdp->buffer))[15]);    
    break;
   
   case USE_LOAD_PS:
   case USE_MUL_PS:
   case USE_ADD_PS:
   case USE_MUL_SS:
   case USE_ADD_SS:
   case USE_DIV_PS:
   case USE_DIV_SS:
   case USE_SQRT_PS:
   case USE_SQRT_SS:
   
   /* avoid FP overflows:
      create x, -1/x, x, -1/x, -x, 1/x, -x, 1/x pattern to guarantee stable register values for add, mul, and mul+add 
      (i.e sum = 0, product = 1, and sum of all partial products = 0) */
   tmp=32*BURST_LENGTH*sizeof(float);
   for (i=0;i<=BUFFERSIZE-tmp;i+=tmp)
   {  
      for(j=0;j<4*BURST_LENGTH;j++)
        *((float*)((unsigned long long)mdp->buffer+i+j*sizeof(float)))=(float)FP_INIT[0];
      for(j=4*BURST_LENGTH;j<8*BURST_LENGTH;j++)
        *((float*)((unsigned long long)mdp->buffer+i+j*sizeof(float)))=-1.0*(float)FP_INIT[1];
      for(j=8*BURST_LENGTH;j<12*BURST_LENGTH;j++)
        *((float*)((unsigned long long)mdp->buffer+i+j*sizeof(float)))=(float)FP_INIT[0];
      for(j=12*BURST_LENGTH;j<16*BURST_LENGTH;j++)
        *((float*)((unsigned long long)mdp->buffer+i+j*sizeof(float)))=-1.0*(float)FP_INIT[1];
      for(j=16*BURST_LENGTH;j<20*BURST_LENGTH;j++)
        *((float*)((unsigned long long)mdp->buffer+i+j*sizeof(float)))=-1.0*(float)FP_INIT[0];
      for(j=20*BURST_LENGTH;j<24*BURST_LENGTH;j++)
        *((float*)((unsigned long long)mdp->buffer+i+j*sizeof(float)))=(float)FP_INIT[1];
      for(j=24*BURST_LENGTH;j<28*BURST_LENGTH;j++)
        *((float*)((unsigned long long)mdp->buffer+i+j*sizeof(float)))=-1.0*(float)FP_INIT[0];
      for(j=28*BURST_LENGTH;j<32*BURST_LENGTH;j++)
        *((float*)((unsigned long long)mdp->buffer+i+j*sizeof(float)))=(float)FP_INIT[1];
   }
   break;
   
   default:
    fprintf( stderr, "Error: initialization failed: unknown mode:%i \n",FUNCTION );      
    exit( 1 );
  }
   //clflush(mdp->buffer,BUFFERSIZE,*(mdp->cpuinfo));
  printf("  wait for threads memory initialization \n");fflush(stdout);
  /* wait for threads to finish their initialization*/  
  for (t=1;t<mdp->num_threads;t++)
  {
     mdp->ack=0;
     mdp->thread_comm[t]=2;
     while (!mdp->ack);
  }
  mdp->ack=0;
  printf("    ...done\n");
  if ((num_packages()!=-1)&&(num_cores_per_package()!=-1)&&(num_threads_per_core()!=-1))  printf("  num_packages: %i, %i cores per package, %i threads per core\n",num_packages(),num_cores_per_package(),num_threads_per_core());
  printf("  using %i threads\n",NUM_THREADS);
  for (i=0;i<NUM_THREADS;i++) if ((get_pkg(cpu_bind[i])!=-1)&&(get_core_id(cpu_bind[i])!=-1)) printf("    - Thread %i runs on CPU %llu, core %i in package: %i\n",i,cpu_bind[i],get_core_id(cpu_bind[i]),get_pkg(cpu_bind[i]));

  #ifdef USE_VTRACE
  VT_USER_END("INIT");
  #endif

  //start watchdog thread
  watchdog_arg.pid=getpid();
  watchdog_arg.timeout=TIMEOUT;
  pthread_create(&watchdog,NULL,watchdog_timer,&watchdog_arg);
  
  return (void*)mdp;
}

/** The central function within each kernel. This function
 *  is called for each measurment step seperately.
 *  @param  mdpv         a pointer to the structure created in bi_init,
 *                       it is the pointer the bi_init returns
 *  @param  problemsize  the actual problemsize
 *  @param  results      a pointer to a field of doubles, the
 *                       size of the field depends on the number
 *                       of functions, there are #functions+1
 *                       doubles
 *  @return 0 if the measurment was sucessfull, something
 *          else in the case of an error
 */
int inline bi_entry( void* mdpv, int problemsize, double* results )
{
  /* j is used for loop iterations */
  int j = 0,k = 0;
  /* real problemsize*/
  unsigned long long rps;
  /* cast void* pointer */
  mydata_t* mdp = (mydata_t*)mdpv;

  /* results */
  double *tmp_results;
  
  #ifdef USE_PAPI
   tmp_results=malloc((1+papi_num_counters)*sizeof(double));
  #else
   tmp_results=malloc(sizeof(double));
  #endif
 
  /* calculate real problemsize */
  rps = problemarray[problemsize-1];

  /* check wether the pointer to store the results in is valid or not */
  if ( results == NULL ) return 1;

  /* call measurement function for */

  _work(rps,OFFSET,FUNCTION,BURST_LENGTH,mdp,&tmp_results);
  results[0] = (double)rps;

  /* copy results */
  results[1]=tmp_results[0];

  #ifdef USE_PAPI
   for (j=0;j<papi_num_counters;j++)
   {
     results[j+2]=tmp_results[j+1];
   }
  #endif
   
  free(tmp_results);
  return 0;
}

/** Clean up the memory
 */
void bi_cleanup( void* mdpv )
{
   int t;
   
   mydata_t* mdp = (mydata_t*)mdpv;
   //terminate other threads
   for (t=1;t<mdp->num_threads;t++)
   {
    mdp->ack=0;
    mdp->thread_comm[t]=0;
    pthread_join((mdp->threads[t]),NULL);
   } 
   pthread_kill(watchdog,SIGUSR1);
   if ( mdp )
   {
     if ((HUGEPAGES==HUGEPAGES_OFF)&&(mdp->buffer)) _mm_free(mdp->buffer);
     if (HUGEPAGES==HUGEPAGES_ON)
     {
       if(mdp->buffer!=NULL) munmap((void*)mdp->buffer,BUFFERSIZE);
     }
     if (mdp->cache_flush_area!=NULL) _mm_free (mdp->cache_flush_area);
     if (mdp->threaddata)
     {
       for (t=1;t<mdp->num_threads;t++)
       {
         if (mdp->threaddata[t].cpuinfo) _mm_free(mdp->threaddata[t].cpuinfo);
       }     
       _mm_free(mdp->threaddata);   
     }
     if (mdp->threads) _mm_free(mdp->threads);
     if (mdp->thread_comm) _mm_free(mdp->thread_comm);
     if (mdp->cpuinfo) _mm_free(mdp->cpuinfo);
     _mm_free( mdp );
   }
   return;
}

/********************************************************************/
/*************** End of interface implementations *******************/
/********************************************************************/

/* Reads the environment variables used by this kernel. */
void evaluate_environment( bi_info * info )
{
   int i;
   char arch[16];
   int errors = 0;
   char * p = 0;
  
   cpuinfo=(cpu_info_t*)_mm_malloc( sizeof( cpu_info_t ),64);
   init_cpuinfo(cpuinfo,1);

   error_msg=malloc(256);
   
   p = bi_getenv( "BENCHIT_KERNEL_PROBLEMLIST", 0 );
   if ( p == 0 )
   {
     int MIN,STEPS;
     double MemFactor;
     p = bi_getenv("BENCHIT_KERNEL_MIN",0);
     if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_MIN not set");}
     else MIN=atoi(p);
     p = bi_getenv("BENCHIT_KERNEL_MAX",0);
     if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_MAX not set");}
     else MAX=atoi(p);
     p = bi_getenv("BENCHIT_KERNEL_STEPS",0);
     if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_STEPS not set");}
     else STEPS=atoi(p);
     if ( errors == 0)
     {
       problemarray=malloc(STEPS*sizeof(double));
       MemFactor =((double)MAX)/((double)MIN);
       MemFactor = pow(MemFactor, 1.0/((double)STEPS-1));
       for (i=0;i<STEPS;i++)
       { 
          problemarray[i] = ((double)MIN)*pow(MemFactor, i);
       }
       problemlistsize=STEPS;
       problemarray[STEPS-1]=(double)MAX;
     }
   }
   else
   {
     fflush(stdout);printf("BenchIT: parsing list of problemsizes: ");
     bi_parselist(p);
     problemlist = info->list;
     problemlistsize = info->listsize;
     problemarray=malloc(problemlistsize*sizeof(double));
     for (i=0;i<problemlistsize;i++)
     { 
        problemarray[i]=problemlist->dnumber;
        if (problemlist->pnext!=NULL) problemlist=problemlist->pnext;
        if (problemarray[i]>MAX) MAX=problemarray[i];
     }
   }
   

   p = bi_getenv( "BENCHIT_KERNEL_CPU_FREQUENCY", 0 );
   if ( p != 0 ) FREQUENCY = atoll( p );
   
   p = bi_getenv( "BENCHIT_KERNEL_L1_SIZE", 0 );
   if ( p != 0 ) 
   {
     L1_SIZE = atoi( p );  
   }

   p = bi_getenv( "BENCHIT_KERNEL_L2_SIZE", 0 );
   if ( p != 0 ) 
   {
     L2_SIZE = atoi( p );  
   }
   
   p = bi_getenv( "BENCHIT_KERNEL_L3_SIZE", 0 );
   if ( p != 0 ) 
   {
     L3_SIZE = atoi( p ); 
   }
   
   p = bi_getenv( "BENCHIT_KERNEL_CACHELINE_SIZE", 0 );
   if ( p != 0 ) CACHELINE = atoi( p );
  
   p = bi_getenv( "BENCHIT_KERNEL_USE_ACCESSES", 0 );
   if ( p != 0 ) NUM_USES = atoi( p );
   else NUM_USES=1;
   
   p = bi_getenv( "BENCHIT_KERNEL_USE_MODE", 0 );
   if ( p == 0 ) USE_MODE=MODE_EXCLUSIVE;
   else
   { 
     if (!strcmp(p,"M")) USE_MODE=MODE_MODIFIED;
     else if (!strcmp(p,"E")) USE_MODE=MODE_EXCLUSIVE;
     else if (!strcmp(p,"I")) USE_MODE=MODE_INVALID;
     else {errors++;sprintf(error_msg,"invalid setting for BENCHIT_KERNEL_USE_MODE");}
   }
   
   CPU_ZERO(&cpuset);NUM_THREADS==0;
   if (bi_getenv( "BENCHIT_KERNEL_CPU_LIST", 0 )!=NULL) p=bi_strdup(bi_getenv( "BENCHIT_KERNEL_CPU_LIST", 0 ));else p=NULL;
   if (p)
   {
     char *q,*r,*s;
     i=0;
     do
     {
       q=strstr(p,",");if (q) {*q='\0';q++;}
       s=strstr(p,"/");if (s) {*s='\0';s++;}
       r=strstr(p,"-");if (r) {*r='\0';r++;}
       
       if ((s)&&(r)) for (i=atoi(p);i<=atoi(r);i+=atoi(s)) {if (cpu_allowed(i)) {CPU_SET(i,&cpuset);NUM_THREADS++;}}
       else if (r) for (i=atoi(p);i<=atoi(r);i++) {if (cpu_allowed(i)) {CPU_SET(i,&cpuset);NUM_THREADS++;}}
       else if (cpu_allowed(atoi(p))) {CPU_SET(atoi(p),&cpuset);NUM_THREADS++;}
       p=q;
     }
     while(p!=NULL);
   }
   else 
   {
     //use all CPUs if not defined otherwise
     for (i=0;i<CPU_SETSIZE;i++) {if (cpu_allowed(i)) {CPU_SET(i,&cpuset);NUM_THREADS++;}}
   }

   if (NUM_THREADS==0) {errors++;sprintf(error_msg,"No allowed CPUs in BENCHIT_KERNEL_CPU_LIST");}
   else
   {
     int j=0;
     cpu_bind=(unsigned long long*)malloc((NUM_THREADS+1)*sizeof(unsigned long long));
     for(i=0;i<CPU_SETSIZE;i++)
     {
      if (CPU_ISSET(i,&cpuset)) {cpu_bind[j]=i;j++;}
     }
   }
   p=bi_getenv( "BENCHIT_KERNEL_ALLOC", 0 );
   if (p==0) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_ALLOC not set");}
   else
   {
     if (!strcmp(p,"G")) ALLOCATION=ALLOC_GLOBAL;
     else if (!strcmp(p,"L")) ALLOCATION=ALLOC_LOCAL;
     else {errors++;sprintf(error_msg,"invalid setting for BENCHIT_KERNEL_ALLOC");}
   }
   
   p=bi_getenv( "BENCHIT_KERNEL_THREAD_OFFSET", 0 );
   if (p!=0)
   {
     THREAD_OFFSET=atoi(p);
   }
   
   p=bi_getenv( "BENCHIT_KERNEL_HUGEPAGES", 0 );
   if (p==0) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_HUGEPAGES not set");}
   else
   {
     if (!strcmp(p,"0")) HUGEPAGES=HUGEPAGES_OFF;
     else if (!strcmp(p,"1")) HUGEPAGES=HUGEPAGES_ON;
     else errors++;
   }
   
   p = bi_getenv( "BENCHIT_KERNEL_OFFSET", 0 );
   if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_OFFSET not set");}
   else OFFSET = atoi( p );
   
   p = bi_getenv( "BENCHIT_KERNEL_BURST_LENGTH", 0 );
   if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_BURST_LENGTH not set");}
   else BURST_LENGTH = atoi( p );
   
   p = bi_getenv( "BENCHIT_KERNEL_ACCESSES", 0 );
   if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_ACCESSES not set");}
   else ACCESSES = atoll( p );
   ACCESSES-=ACCESSES%96; /* 32 or 48 inner loop length */
   if (ACCESSES<MAX) {errors++;sprintf(error_msg,"ACCESSES has to be larger than biggest data set size");}
   
   p = bi_getenv( "BENCHIT_KERNEL_USE_DIRECTION", 0 );
   if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_USE_DIRECTION not set");}
   else
   {
     if (!strcmp(p,"LIFO")) USE_DIRECTION=LIFO;
     else if (!strcmp(p,"FIFO")) USE_DIRECTION=FIFO;
     else {errors++;sprintf(error_msg,"invalid setting for BENCHIT_KERNEL_USE_DIRECTION");}
   }
   
   p=bi_getenv( "BENCHIT_KERNEL_INSTRUCTION", 0 );
   if (p==0) errors++;
   else
   {
     if (!strcmp(p,"load_pi")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_LOAD_PI;}
     else if (!strcmp(p,"load_pd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_LOAD_PD;}
     else if (!strcmp(p,"load_ps")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_LOAD_PS;}
     else if (!strcmp(p,"store")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_STORE;}
     else if (!strcmp(p,"store_nt")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_STORE_NT;}
     else if (!strcmp(p,"copy")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_COPY;}
     else if (!strcmp(p,"copy_nt")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_COPY_NT;}
     else if (!strcmp(p,"scale_int")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_SCALE_INT;}
     else if (!strcmp(p,"mul_pi")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_MUL_PI;}
     else if (!strcmp(p,"add_pi")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_ADD_PI;}
     else if (!strcmp(p,"mul_si")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_MUL_SI;}
     else if (!strcmp(p,"add_si")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_ADD_SI;}
     else if (!strcmp(p,"mul_pd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_MUL_PD;}
     else if (!strcmp(p,"add_pd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_ADD_PD;}
     else if (!strcmp(p,"mul_sd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_MUL_SD;}
     else if (!strcmp(p,"add_sd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_ADD_SD;}
     else if (!strcmp(p,"mul_ps")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_MUL_PS;}
     else if (!strcmp(p,"add_ps")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_ADD_PS;}
     else if (!strcmp(p,"mul_ss")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_MUL_SS;}
     else if (!strcmp(p,"add_ss")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_ADD_SS;}
     else if (!strcmp(p,"div_pd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_DIV_PD;}
     else if (!strcmp(p,"div_ps")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_DIV_PS;}
     else if (!strcmp(p,"div_sd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_DIV_SD;}
     else if (!strcmp(p,"div_ss")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_DIV_SS;}
     else if (!strcmp(p,"sqrt_pd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_SQRT_PD;}
     else if (!strcmp(p,"sqrt_ps")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_SQRT_PS;}
     else if (!strcmp(p,"sqrt_sd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_SQRT_SD;}
     else if (!strcmp(p,"sqrt_ss")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_SQRT_SS;}
     else if (!strcmp(p,"and_pd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_AND_PD;}
     else if (!strcmp(p,"and_pi")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_AND_PI;}
     else if (!strcmp(p,"mul_add_pd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_MUL_ADD_PD;}
     else if (!strcmp(p,"mul+add_pd")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_MUL_PLUS_ADD_PD;}
     else {errors++;ALIGNMENT=128;sprintf(error_msg,"invalid setting for BENCHIT_KERNEL_INSTRUCTION");}
   }

   p=bi_getenv("BENCHIT_KERNEL_INT_INIT", 0);
   if (p)
   {
      INT_INIT[0]=atoll(p);
      INT_INIT[1]=INT_INIT[0]*-1;
   }
   
   p=bi_getenv("BENCHIT_KERNEL_FP_INIT", 0);
   if (p)
   {
      FP_INIT[0]=atof(p);
      FP_INIT[1]=(1.0/FP_INIT[0]);
   }

   mdp = (mydata_t*)_mm_malloc( sizeof( mydata_t ),ALIGNMENT);
   if ( mdp == 0 )
   {
      fprintf( stderr, "Error: Allocation of structure mydata_t failed\n" ); fflush( stderr );
      exit( 127 );
   }
   memset(mdp,0,sizeof(mydata_t));

   p=bi_getenv( "BENCHIT_KERNEL_TIMEOUT", 0 );
   if (p!=0)
   {
     TIMEOUT=atoi(p);
   }
   
   #ifdef USE_PAPI
   p=bi_getenv( "BENCHIT_KERNEL_ENABLE_PAPI", 0 );
   if (p==0) errors++;
   else if (atoi(p)==1)
   {
      papi_num_counters=0;
      p=bi_getenv( "BENCHIT_KERNEL_PAPI_COUNTERS", 0 );
      if ((p!=0)&&(strcmp(p,"")))
      {
        if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
        {
          printf("Warning: PAPI library init error\n");errors++;
        }
        else
        {      
          char* tmp;
          papi_num_counters=1;
          tmp=p;
          PAPI_thread_init(pthread_self);
          while (strstr(tmp,",")!=NULL) {tmp=strstr(tmp,",")+1;papi_num_counters++;}
          papi_names=(char**)malloc(papi_num_counters*sizeof(char*));
          papi_codes=(int*)malloc(papi_num_counters*sizeof(int));
         
          tmp=p;
          for (i=0;i<papi_num_counters;i++)
          {
            tmp=strstr(tmp,",");
            if (tmp!=NULL) {*tmp='\0';tmp++;}
            papi_names[i]=p;p=tmp;
            if (PAPI_event_name_to_code(papi_names[i],&papi_codes[i])!=PAPI_OK)
            {
             sprintf(error_msg,"Papi error: unknown Counter: %s\n",papi_names[i]);
             papi_num_counters=0;errors++;
            }
          }
          
          EventSet = PAPI_NULL;
          if (PAPI_create_eventset(&EventSet) != PAPI_OK)
          {
             sprintf(error_msg,"PAPI error, could not create eventset\n");
             papi_num_counters=0;errors++;
          }
          for (i=0;i<papi_num_counters;i++)
          { 
            if ((PAPI_add_event(EventSet, papi_codes[i]) != PAPI_OK))
            {
              sprintf(error_msg,"PAPI error, could add counter %s to eventset.\n",papi_names[i]);
              papi_num_counters=0;errors++;
            }
          }

          /*printf("num_counters: %i\n",papi_num_counters);
          for (i=0;i<papi_num_counters;i++)
          {
             printf("PAPI counter %i: %s(%08x)\n",i,papi_names[i],papi_codes[i]);     
          }
          fflush(stdout);*/
        }
      }
      #ifdef USE_VTRACE
       mdp->cid_papi=calloc(papi_num_counters,sizeof(unsigned int));
       mdp->gid_papi=calloc(papi_num_counters,sizeof(unsigned int));
       for (i=0;i<papi_num_counters;i++)
       {
         mdp->gid_papi[i] = VT_COUNT_GROUP_DEF("papi counters");
         mdp->cid_papi[i] = VT_COUNT_DEF(papi_names[i], "number", VT_COUNT_TYPE_DOUBLE, mdp->gid_papi[i]);
       }
      #endif
      if (papi_num_counters>0) PAPI_start(EventSet);
   }
   #endif
   
   #ifdef UNCORE
   p=bi_getenv( "BENCHIT_KERNEL_ENABLE_NEHALEM_UNCORE_EVENTS", 0 );
   if (p==0) errors++;
   else if (atoi(p)==1)
   {
      mdp->pfmon_num_events=0;
      p=bi_getenv( "BENCHIT_KERNEL_UNCORE_EVENT", 0 );
      if ((p!=0)&&(strcmp(p,"")))
      {
        if (pfm_initialize() != PFMLIB_SUCCESS) {
         printf("can not initialize libpfm\n");
         errors++;
        }
        else
        {     
          char* tmp;
          mdp->pfmon_num_events=1;
          tmp=p;
          while (strstr(tmp,",")!=NULL) {tmp=strstr(tmp,",")+1;mdp->pfmon_num_events++;}

          mdp->pfm_names=(char**)malloc(mdp->pfmon_num_events*sizeof(char*));
          mdp->pfm_codes=(int*)malloc(mdp->pfmon_num_events*sizeof(int));

          
          memset(&mdp->inp,0, sizeof(mdp->inp));
          memset(&mdp->outp,0, sizeof(mdp->outp));
          
          tmp=p;
          for (i=0;i<mdp->pfmon_num_events;i++)
          {
            tmp=strstr(tmp,",");
            if (tmp!=NULL) {*tmp='\0';tmp++;}
            mdp->pfm_names[i]=p;p=tmp;

            if (pfm_find_full_event(mdp->pfm_names[i], &mdp->inp.pfp_events[i]) != PFMLIB_SUCCESS)
            {
              printf("Warning: cannot find event: %s\n");
              mdp->pfmon_num_events=0;errors++;
            }
            mdp->inp.pfp_events[i].plm=PFM_PLM3|PFM_PLM0;
          }
          mdp->inp.pfp_dfl_plm     = PFM_PLM3|PFM_PLM0;
          mdp->inp.pfp_flags       = PFMLIB_PFP_SYSTEMWIDE;
          mdp->inp.pfp_event_count = mdp->pfmon_num_events;
          #ifdef USE_VTRACE
          mdp->cid_pfm=calloc(mdp->pfmon_num_events,sizeof(unsigned int));
          mdp->gid_pfm=calloc(mdp->pfmon_num_events,sizeof(unsigned int));
          for (i=0;i<mdp->pfmon_num_events;i++)
          {
            mdp->gid_pfm[i] = VT_COUNT_GROUP_DEF("perfmon2 uncore events");
            mdp->cid_pfm[i] = VT_COUNT_DEF(mdp->pfm_names[i], "number", VT_COUNT_TYPE_DOUBLE, mdp->gid_pfm[i]);
          }
          #endif
        }
      }
   }
   #endif

   if ((BURST_LENGTH>4)&&(BURST_LENGTH!=8)) {errors++;sprintf(error_msg,"BURST LENGTH %i not supported",BURST_LENGTH);}
   
   if ( errors > 0 )
   {
      fprintf( stderr, "Error: There's an environment variable not set or invalid!\n" );      
      fprintf( stderr, "%s\n", error_msg);
      exit( 1 );
   }
   
   get_architecture(arch,sizeof(arch));
   if (strcmp(arch,"x86_64")) {
      fprintf( stderr, "Error: wrong architecture: %s, x86_64 required \n",arch );
      exit( 1 );
   }
   
   if (!feature_available("CLFLUSH")) {
      fprintf( stderr, "Error: required function \"clflush\" not supported!\n" );
      exit( 1 );
   }
   
   if (!feature_available("CPUID")) {
      fprintf( stderr, "Error: required function \"cpuid\" not supported!\n" );
      exit( 1 );
   }
   
   if (!feature_available("TSC")) {
      fprintf( stderr, "Error: required function \"rdtsc\" not supported!\n" );
      exit( 1 );
   }
   
   if (!feature_available("SSE2")) {
        fprintf( stderr, "Error: SSE2 not supported!\n" );
        exit( 1 );
   }

   switch (FUNCTION)
   {
     case USE_MUL_PI:
      if (!feature_available("SSE4.1")) {
        fprintf( stderr, "Error: SSE4.1 not supported!\n" );
        exit( 1 );
      }
     default:
       break;
   }
}
