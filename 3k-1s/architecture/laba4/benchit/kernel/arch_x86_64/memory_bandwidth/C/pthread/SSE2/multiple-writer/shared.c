/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
 /* identical functions for all multiple worker bandwidth kernels */
 
#include <pthread.h>
#include "shared.h"
#include "interface.h"

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>

#ifdef USE_PAPI
#include <papi.h>
#endif

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
 * with respect to reduced capacity of caches, that are shared between threads (NOTE: worst case assumptions are
 * made, i.e flushes might start at smaller memsizes than requiered [e.g] when running 2 threads on a 2 socket
 * system with dual core processors it is assumed that both threads are running on a single package)
 * TODO: use cache shared map from sysfs and cpu bindings to figure out what is actally shared
 */
void inline flush_caches(void* buffer,unsigned long long memsize,int settings,int num_flushes,int flush_mode,void* flush_buffer,cpu_info_t *cpuinfo,int id,int num_threads)
{
   int i,j,cachesize;
   for (i=cpuinfo->Cachelevels;i>0;i--)
   {   
     if (settings&FLUSH(i))
     {
       cachesize=cpuinfo->U_Cache_Size[i-1]+cpuinfo->D_Cache_Size[i-1];
       if (cpuinfo->Cache_shared[i-1]>1)
       { 
          if (num_threads<=cpuinfo->Cache_shared[i-1]) cachesize/=num_threads;
          else cachesize/=cpuinfo->Cache_shared[i-1];
       }
       /* add size of lower cache levels in case of non-inclusive caches*/
       /* TODO num threads * per core caches*/
       if (!strcmp(cpuinfo->vendor,"AuthenticAMD"))
       {
         for (j=i-1;j>0;j--)
         {
           cachesize+=cpuinfo->U_Cache_Size[j-1]+cpuinfo->D_Cache_Size[j-1];
         }
       }
       if (memsize>cachesize)
       {
         if (cpuinfo->Cachelevels>i)cacheflush(i,num_flushes,flush_mode,flush_buffer,*(cpuinfo));
         else if (clflush( buffer , memsize, *(cpuinfo) )) cacheflush(i,num_flushes,flush_mode,flush_buffer,*(cpuinfo));
         break;
       }
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
    mydata->aligned_addr=(unsigned long long)(mydata->buffer) + mydata->offset + id*mydata->thread_offset;
  }  
  
  while(1)
  {
     //printf("Thread %i: comm= %i\n",id+1,data->thread_comm[id]);
     switch (global_data->thread_comm[id]){
       case THREAD_USE_MEMORY: 
         if (old!=THREAD_USE_MEMORY)
         {
           global_data->ack=id;old=THREAD_USE_MEMORY;
           
           if (!mydata->buffersize)
           {
             mydata->buffer = (char*) (((unsigned long long)global_data->buffer+((id)*mydata->memsize)+mydata->alignment)&((mydata->alignment-1)^0xffffffffffffffffULL));
             mydata->aligned_addr = (unsigned long long)(mydata->buffer) + mydata->offset;
           }
           
           // use memory
           use_memory((void*)mydata->aligned_addr,mydata->memsize,mydata->USE_MODE,mydata->USE_DIRECTION,mydata->NUM_USES,*(mydata->cpuinfo));
 
           //flush cachelevels as specified in PARAMETERS
           flush_caches((void*) (mydata->aligned_addr),mydata->memsize,mydata->settings,mydata->NUM_FLUSHES,mydata->FLUSH_MODE,mydata->cache_flush_area,mydata->cpuinfo,id,global_data->running_threads);
           
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
       case THREAD_WORK:
          if (old!=THREAD_WORK) 
          {
            global_data->ack=id;old=THREAD_WORK;
            
            if (!mydata->buffersize)
            {
             mydata->buffer = (char*) (((unsigned long long)global_data->buffer+((id)*mydata->memsize)+mydata->alignment)&((mydata->alignment-1)^0xffffffffffffffffULL));
             mydata->aligned_addr = (unsigned long long)(mydata->buffer) + mydata->offset;
            }
            //printf("Thread %i, address: %lu\n",id,mydata->aligned_addr);
            
            /* call ASM implementation */
            switch (mydata->FUNCTION)
            {
               case USE_MOVNTDQ: tmp=asm_work_movntdq(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_MOVDQA: tmp=asm_work_movdqa(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_MOVDQU: tmp=asm_work_movdqu(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                              
               case USE_MOVNTI: tmp=asm_work_movnti(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;               
               case USE_MOV: tmp=asm_work_mov(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;               
               default: break;
            }
            global_data->done=id;
          }
          else 
          {
            tmp=100;while(tmp>0) tmp--; 
          }  
          break;
       case THREAD_INIT: // used for parallel initialisation only
          tmp=100;while(tmp) tmp--; 
          break;
       case THREAD_STOP: // exit
       default:
         {
           if(mydata->buffer!=NULL) munmap((void*)mydata->buffer,mydata->buffersize);
         }
         pthread_exit(NULL);
    }
  }
}
