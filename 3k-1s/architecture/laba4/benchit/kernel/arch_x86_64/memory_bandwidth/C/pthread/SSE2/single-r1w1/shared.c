/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures combined bandwidth of one read and one write stream located
 *         in different cache levels or memory of certain CPUs.
 *******************************************************************/
 
#include <pthread.h>
#include "shared.h"
#include "interface.h"

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>

/*
 * flush caches
 */
void inline flush_caches(param_t *params)
{
  register unsigned long long tmp=0;

  if ((params->flush_mode==MODE_MODIFIED)||(params->flush_mode==MODE_EXCLUSIVE)||(params->flush_mode==MODE_INVALID))
  { 
   params->j=params->num_flushes;
   while(params->j--)
   {
    for (params->i=0;params->i<params->flushsize;params->i+=STRIDE)
    {
      //tmp=*((unsigned long long*)((unsigned long long)(params->flushaddr)+params->i)); //needed if non-destructive operation is required
      *((unsigned long long*)((unsigned long long)(params->flushaddr)+params->i))=tmp;
    }
   }
  }
  if (params->flush_mode==MODE_EXCLUSIVE) 
  {
    __asm__ __volatile__("mfence;"::);
    for(params->i = params->flushsize/64;params->i>=0;params->i--)
    {
      __asm__ __volatile__("clflush (%%rax);":: "a" (((unsigned long long)params->flushaddr)+64*params->i));
    }
    __asm__ __volatile__("mfence;"::);
    params->j=params->num_flushes;
    while(params->j--)
    {
     for (params->i=0;params->i<params->flushsize;params->i+=STRIDE)
     {
       tmp|=*((unsigned long long*)((unsigned long long)(params->flushaddr)+params->i));
     }
     /* store result somewhere to prevent compiler from "optimizations" */
     *((unsigned long long*)((unsigned long long)(params->flushaddr)+params->i))=tmp;
    }
  }
  if (params->flush_mode==MODE_INVALID) 
  {
    __asm__ __volatile__("mfence;"::);
    for(params->i = params->flushsize/64;params->i>=0;params->i--)
    {
      __asm__ __volatile__("clflush (%%rax);":: "a" (((unsigned long long)params->flushaddr)+64*params->i));
    }
    __asm__ __volatile__("mfence;"::);
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
  param_t* params;

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
      *((double*)((unsigned long long)mydata->buffer+i))=(double)i;
    }
    clflush(mydata->buffer,mydata->buffersize,*(mydata->cpuinfo));
    mydata->aligned_addr=(unsigned long long)(mydata->buffer) + mydata->offset;
  }
  else mydata->aligned_addr=(unsigned long long)(global_data->buffer) + mydata->offset;
  
  params=(param_t*)(mydata->aligned_addr + global_data->buffersize/2 + global_data->offset);
  
  while(1)
  {
     //printf("Thread %i: comm= %i\n",id+1,data->thread_comm[id]);
     switch (global_data->thread_comm[id]){
       case THREAD_USE_MEMORY: 
         if (old!=THREAD_USE_MEMORY)
         {
           global_data->ack=id;old=THREAD_USE_MEMORY;
           /* use memory */           
           use_memory(params);
 
           /* flush cachelevels as specified in PARAMETERS */
           if (params->flushsize) flush_caches(params);
           
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
