/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures read bandwidth of data located in different cache
 *         levels or memory of certain CPUs.
 *******************************************************************/
 /* identical functions for all single threaded kernels 
  * TODO: migrate changes from r1w1 kernel */
 
#include <pthread.h>
#include "shared.h"
#include "interface.h"

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>

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
           // use memory
           use_memory((void*)mydata->aligned_addr,mydata->memsize,mydata->USE_MODE,mydata->USE_DIRECTION,mydata->NUM_USES,*(mydata->cpuinfo));
 
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
