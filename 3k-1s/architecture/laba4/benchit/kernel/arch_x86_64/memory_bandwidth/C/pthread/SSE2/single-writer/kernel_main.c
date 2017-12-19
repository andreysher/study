/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/** Kernel: measures write bandwidth of data located in different cache
 *         levels or memory of certain CPUs.
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
#include <signal.h>
#include "interface.h"

/*  Header for local functions
 */
#include "work.h"
#include "shared.h"

#ifdef USE_PAPI
#include <papi.h>
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
int ALLOCATION, HUGEPAGES, RUNS, OFFSET=0, BUFFERSIZE, FUNCTION;
int NUM_FLUSHES=0,NUM_USES=0,FLUSH_MODE,USE_MODE,NUM_THREADS=0,SHARE_CPU;
int FLUSH_L1=0,FLUSH_L2=0,FLUSH_L3=0,ALIGNMENT,BURST_LENGTH,USE_DIRECTION;
int NUM_RESULTS,TIMEOUT=0;

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
   (void) memset ( infostruct, 0, sizeof( bi_info ) );
   /* get environment variables for the kernel */
   evaluate_environment(infostruct);
   infostruct->codesequence = bi_strdup( "mov xmm -> mem" );
   infostruct->xaxistext = bi_strdup( "data set size [Byte]" );
   infostruct->base_xaxis=10.0;
   infostruct->num_measurements=problemlistsize;
   sprintf(buff,"single threaded memory bandwidth");
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
   
   /* measure local bandwidth of CPU0 and bandwidth between CPU0 and all other selected CPUs*/
   n_of_sure_funcs_per_work = NUM_RESULTS;
   
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
   *infostruct->selected_result=0;
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
      for (k=0;k<NUM_THREADS;k++)
      {
         sprintf(buf,"memory bandwidth: CPU%llu writing memory used by CPU%llu",cpu_bind[0],cpu_bind[k]);
         index= k + n_of_sure_funcs_per_work * j;
         infostruct->yaxistexts[index] = bi_strdup( "bandwidth [GB/s]" );
         infostruct->selected_result[index] = 1;
         infostruct->base_yaxis[index] = 0;
         switch ( j )
        {
          case 0:
            infostruct->legendtexts[index] = bi_strdup( buf );
          break;
          default: // papi
           #ifdef USE_PAPI
            if (k)  sprintf(buff,"%s CPU%llu - CPU%llu",papi_names[j-1],cpu_bind[0],cpu_bind[k]);
            else sprintf(buff,"%s CPU%llu locally",papi_names[j-1],cpu_bind[0]);
            infostruct->legendtexts[index] = bi_strdup( buff );
            infostruct->selected_result[index] = 0;
            infostruct->yaxistexts[index] = bi_strdup( "counter value/ memory accesses" );
           #endif
           break;
        } 
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
   mydata_t* mdp;
   int retval,t,i;
   long long tmp;

/*
   printf("\n");   
   printf("sizeof mydata_t:           %i\n",sizeof(mydata_t));
   printf("sizeof threaddata_t:       %i\n",sizeof(threaddata_t));
   printf("sizeof cpu_info_t:         %i\n",sizeof(cpu_info_t));
*/

  cpu_set(cpu_bind[0]);

   mdp = (mydata_t*)_mm_malloc( sizeof( mydata_t ),ALIGNMENT);
   if ( mdp == 0 )
   {
      fprintf( stderr, "Error: Allocation of structure mydata_t failed\n" ); fflush( stderr );
      exit( 127 );
   }
   memset(mdp,0,sizeof(mydata_t));
   
   mdp->cpuinfo=cpuinfo;

   BUFFERSIZE=sizeof(char)*(MAX+ALIGNMENT+OFFSET+2*sizeof(unsigned long long));
   if (HUGEPAGES==HUGEPAGES_ON) BUFFERSIZE=(BUFFERSIZE+(2*1024*1024))&0xffe00000ULL;
   
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

   mdp->NUM_FLUSHES=NUM_FLUSHES;
   mdp->NUM_USES=NUM_USES;
   mdp->FLUSH_MODE=FLUSH_MODE;
   mdp->USE_MODE=USE_MODE;
   mdp->SHARE_CPU=SHARE_CPU;
   mdp->USE_DIRECTION=USE_DIRECTION;
   
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
   
   if ((FLUSH_L1)&&(mdp->cpuinfo->U_Cache_Size[0]+mdp->cpuinfo->D_Cache_Size[0]!=0))
   { 
      mdp->settings|=FLUSH(1);
      if (mdp->cpuinfo->Cacheline_size[0]==0)
      {
        fprintf( stderr, "Error: unknown Cacheline-length\n" );
        exit( 1 );    
      }     
      CACHEFLUSHSIZE=mdp->cpuinfo->U_Cache_Size[0];
      CACHEFLUSHSIZE+=mdp->cpuinfo->D_Cache_Size[0];
   }
   if ((FLUSH_L2)&&(mdp->cpuinfo->U_Cache_Size[1]+mdp->cpuinfo->D_Cache_Size[1]!=0))
   {
      mdp->settings|=FLUSH(2);
      if (mdp->cpuinfo->Cacheline_size[1]==0)
      {
        fprintf( stderr, "Error: unknown Cacheline-length\n" );
        exit( 1 );    
      }     
      CACHEFLUSHSIZE=mdp->cpuinfo->U_Cache_Size[1]+mdp->cpuinfo->U_Cache_Size[0];
      CACHEFLUSHSIZE+=mdp->cpuinfo->D_Cache_Size[1]+mdp->cpuinfo->D_Cache_Size[0];
   }
   if ((FLUSH_L3)&&(mdp->cpuinfo->U_Cache_Size[2]+mdp->cpuinfo->D_Cache_Size[2]!=0))
   { 
      mdp->settings|=FLUSH(3);
      if (mdp->cpuinfo->Cacheline_size[2]==0)
      {
        fprintf( stderr, "Error: unknown Cacheline-length\n" );
        exit( 1 );    
      }     
      CACHEFLUSHSIZE=mdp->cpuinfo->U_Cache_Size[2]+mdp->cpuinfo->U_Cache_Size[1]+mdp->cpuinfo->U_Cache_Size[0];
      CACHEFLUSHSIZE+=mdp->cpuinfo->D_Cache_Size[2]+mdp->cpuinfo->D_Cache_Size[1]+mdp->cpuinfo->D_Cache_Size[0];
   }
   
   CACHEFLUSHSIZE*=14;
   CACHEFLUSHSIZE/=10;

   if (CACHEFLUSHSIZE>mdp->cpuinfo->Cacheflushsize)
   {
      //printf("overwriting Cacheflushsize\n");
      mdp->cpuinfo->Cacheflushsize=CACHEFLUSHSIZE;
   }

   if (CACHELEVELS>mdp->cpuinfo->Cachelevels)
   {
      //printf("overwriting Cachelevels\n");
      mdp->cpuinfo->Cachelevels=CACHELEVELS;
   }

   if ((FLUSH_L1)||(FLUSH_L2)||(FLUSH_L3))
   {
     mdp->cache_flush_area=(char*)_mm_malloc(mdp->cpuinfo->Cacheflushsize,ALIGNMENT);
     if (mdp->cache_flush_area == 0)
     {
        fprintf( stderr, "Error: Allocation of structure mydata_t failed\n" ); fflush( stderr );
        exit( 127 );
     }
     //fill cacheflush-area
     tmp=sizeof(unsigned long long);
     for (i=0;i<mdp->cpuinfo->Cacheflushsize;i+=tmp)
     {
        *((unsigned long long*)((unsigned long long)mdp->cache_flush_area+i))=(unsigned long long)i;
     }
     clflush(mdp->cache_flush_area,mdp->cpuinfo->Cacheflushsize,*(mdp->cpuinfo));
   }
   else mdp->cpuinfo->Cacheflushsize=0;     
 
   printf("\n");
     
   if (mdp->settings&FLUSH(1)) printf("  enabled L1 flushes\n");
   if (mdp->settings&FLUSH(2)) printf("  enabled L2 flushes\n");
   if (mdp->settings&FLUSH(3)) printf("  enabled L3 flushes\n");

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
  
  // create threads
  for (t=1;t<mdp->num_threads;t++)
  {
    mdp->threaddata[t].cpuinfo=(cpu_info_t*)_mm_malloc( sizeof( cpu_info_t ),ALIGNMENT);
    if ( mdp->cpuinfo == 0 )
    {
      fprintf( stderr, "Error: Allocation of structure mydata_t failed\n" ); fflush( stderr );
      exit( 127 );
    }
    memcpy(mdp->threaddata[t].cpuinfo,mdp->cpuinfo,sizeof(cpu_info_t));
    mdp->ack=0;
    mdp->threaddata[t].thread_id=t;
    mdp->threaddata[t].cpu_id=cpu_bind[t];
    mdp->threaddata[t].data=mdp;
    mdp->thread_comm[t]=THREAD_INIT;
    mdp->threaddata[t].settings=mdp->settings;
    /*mdp->threaddata[t].cache_flush_area=mdp->cache_flush_area;*/
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
    mdp->threaddata[t].USE_DIRECTION=mdp->USE_DIRECTION;
    mdp->threaddata[t].NUM_FLUSHES=mdp->NUM_FLUSHES;
    mdp->threaddata[t].FLUSH_MODE=mdp->FLUSH_MODE;
    if (ALLOCATION==ALLOC_LOCAL) mdp->threaddata[t].buffersize=BUFFERSIZE;
    if (ALLOCATION==ALLOC_GLOBAL) mdp->threaddata[t].buffersize=0;
    mdp->threaddata[t].alignment=ALIGNMENT;
    mdp->threaddata[t].offset=OFFSET;    
    
    pthread_create(&(mdp->threads[t]),NULL,thread,(void*)(&(mdp->threaddata[t])));
    while (!mdp->ack);
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
   tmp=sizeof(unsigned long long);
   for (i=0;i<=BUFFERSIZE-tmp;i+=tmp)
   {
      *((unsigned long long*)((unsigned long long)mdp->buffer+i))=(unsigned long long)i;
   }
   clflush(mdp->buffer,BUFFERSIZE,*(mdp->cpuinfo));

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


  _work(rps,OFFSET,FUNCTION,BURST_LENGTH,RUNS,mdp,&tmp_results);
  results[0] = (double)rps;

  /* copy tmp_results to final results */  
  for (k=0;k<NUM_RESULTS;k++)
  {
    results[1+k]=tmp_results[k];
    #ifdef USE_PAPI
    for (j=0;j<papi_num_counters;j++)
    {
      results[1+(j+1)*NUM_RESULTS+k]=mdp->papi_results[j];
    }
    #endif
  }
    
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
void evaluate_environment(bi_info * info)
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
   NUM_RESULTS=NUM_THREADS;

   p = bi_getenv( "BENCHIT_KERNEL_CPU_FREQUENCY", 0 );
   if ( p != 0 ) FREQUENCY = atoll( p );

   p = bi_getenv( "BENCHIT_KERNEL_RUNS", 0 );
   if ( p != 0 ) RUNS = atoi( p );
   else RUNS=1;

   p = bi_getenv( "BENCHIT_KERNEL_FLUSH_L1", 0 );
   if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_FLUSH_L1 not set");}
   else FLUSH_L1 = atoi( p );
   p = bi_getenv( "BENCHIT_KERNEL_FLUSH_L2", 0 );
   if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_FLUSH_L2 not set");}
   else FLUSH_L2 = atoi( p );
   p = bi_getenv( "BENCHIT_KERNEL_FLUSH_L3", 0 );
   if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_FLUSH_L3 not set");}
   else FLUSH_L3 = atoi( p );
   
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


   p = bi_getenv( "BENCHIT_KERNEL_FLUSH_ACCESSES", 0 );
   if ( p != 0 ) NUM_FLUSHES = atoi( p );
   else NUM_FLUSHES=1;

   p = bi_getenv( "BENCHIT_KERNEL_FLUSH_MODE", 0 );
   if ( p == 0 ) FLUSH_MODE=MODE_EXCLUSIVE;
   else
   { 
     if (!strcmp(p,"M")) FLUSH_MODE=MODE_MODIFIED;
     else if (!strcmp(p,"E")) FLUSH_MODE=MODE_EXCLUSIVE;
     else if (!strcmp(p,"I")) FLUSH_MODE=MODE_INVALID;
     else {errors++;sprintf(error_msg,"invalid setting for BENCHIT_KERNEL_FLUSH_MODE");}
   }

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
     else if (!strcmp(p,"S")) USE_MODE=MODE_SHARED;
     else if (!strcmp(p,"O")) USE_MODE=MODE_OWNED;
     else if (!strcmp(p,"F")) USE_MODE=MODE_FORWARD;
     else {errors++;sprintf(error_msg,"invalid setting for BENCHIT_KERNEL_USE_MODE");}
     if ((USE_MODE==MODE_SHARED)||(USE_MODE==MODE_OWNED)||(USE_MODE==MODE_FORWARD))
     {
       p = bi_getenv( "BENCHIT_KERNEL_SHARE_CPU", 0 );
       SHARE_CPU=0;
       if (p==0) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_SHARE_CPU not set");}
       else {
          SHARE_CPU=atoi(p);
          if (CPU_ISSET(SHARE_CPU,&cpuset)) {errors++;sprintf(error_msg,"SHARE_CPU must not be used for measurement");}
          NUM_THREADS++;
          cpu_bind[NUM_THREADS-1]=SHARE_CPU;
          SHARE_CPU=NUM_THREADS-1;
       }
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
   
   p=bi_getenv( "BENCHIT_KERNEL_HUGEPAGES", 0 );
   if (p==0) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_HUGEPAGES not set");}
   else
   {
     if (!strcmp(p,"0")) HUGEPAGES=HUGEPAGES_OFF;
     else if (!strcmp(p,"1")) HUGEPAGES=HUGEPAGES_ON;
     else {errors++;sprintf(error_msg,"invalid setting for BENCHIT_KERNEL_HUGEPAGES");}
   }

   p = bi_getenv( "BENCHIT_KERNEL_OFFSET", 0 );
   if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_OFFSET not set");}
   else OFFSET = atoi( p );
   
   p = bi_getenv( "BENCHIT_KERNEL_BURST_LENGTH", 0 );
   if ( p == 0 ) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_BURST_LENGTH not set");}
   else BURST_LENGTH = atoi( p );
   
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
     if (!strcmp(p,"mov")) {ALIGNMENT=64;OFFSET=OFFSET%ALIGNMENT;FUNCTION=USE_MOV;}
     else if (!strcmp(p,"movnti")) {ALIGNMENT=64;OFFSET=OFFSET%ALIGNMENT;FUNCTION=USE_MOVNTI;}
     else if (!strcmp(p,"movdqa")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_MOVDQA;}
     else if (!strcmp(p,"movdqu")) {ALIGNMENT=128;OFFSET=OFFSET%ALIGNMENT;FUNCTION=USE_MOVDQU;}
     else if (!strcmp(p,"movntdq")) {ALIGNMENT=128;OFFSET=0;FUNCTION=USE_MOVNTDQ;}
     else {errors++;sprintf(error_msg,"invalid setting for BENCHIT_KERNEL_INSTRUCTION");}
   }
   
   p=bi_getenv( "BENCHIT_KERNEL_TIMEOUT", 0 );
   if (p!=0)
   {
     TIMEOUT=atoi(p);
   }
   
   p=bi_getenv( "BENCHIT_KERNEL_SERIALIZATION", 0 );
   if ((p!=0)&&(strcmp(p,"mfence"))&&(strcmp(p,"cpuid"))) {errors++;sprintf(error_msg,"invalid setting for BENCHIT_KERNEL_SERIALIZATION");}
   
   #ifdef USE_PAPI
   p=bi_getenv( "BENCHIT_KERNEL_ENABLE_PAPI", 0 );
   if (p==0) {errors++;sprintf(error_msg,"BENCHIT_KERNEL_ENABLE_PAPI not set");}
   else if (atoi(p)==1)
   {
      papi_num_counters=0;
      p=bi_getenv( "BENCHIT_KERNEL_COUNTERS", 0 );
      if ((p!=0)&&(strcmp(p,"")))
      {
        if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
        {
          sprintf(error_msg,"PAPI library init error\n");errors++;
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
             sprintf(error_msg,"Papi error: unknown Counter: %s\n",papi_names[i]);fflush(stdout);
             papi_num_counters=0;errors++;
            }
          }
          
          EventSet = PAPI_NULL;
          if (PAPI_create_eventset(&EventSet) != PAPI_OK)
          {
             sprintf(error_msg,"PAPI error, could not create eventset\n");fflush(stdout);
             papi_num_counters=0;errors++;
          }
          for (i=0;i<papi_num_counters;i++)
          { 
            if ((PAPI_add_event(EventSet, papi_codes[i]) != PAPI_OK))
            {
              sprintf(error_msg,"PAPI error, could add counter %s to eventset.\n",papi_names[i]);fflush(stdout);
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
      if (papi_num_counters>0) PAPI_start(EventSet);
   }
   #endif
   
   if (BURST_LENGTH>4) {errors++;sprintf(error_msg,"BURST LENGTH %i not supported",BURST_LENGTH);}
   
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
   
   switch (FUNCTION)
   {
     case USE_MOVDQA:
     case USE_MOVDQU:
     case USE_MOVNTDQ:
      if (!feature_available("SSE2")) {
        fprintf( stderr, "Error: SSE2 not supported!\n" );
        exit( 1 );
      }
     default:
       break;
   }
}
