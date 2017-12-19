/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: membw.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/bandwidth/C/MPI/0/AeApBxC/membw.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Memory Bandwidth (C)
 *******************************************************************/

#include "interface.h"
#include "membw.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#ifndef BENCHIT_KERNEL_MIN_ACCESS_SIZE
#define BENCHIT_KERNEL_MIN_ACCESS_SIZE (2048)
#endif

#ifndef BENCHIT_KERNEL_MAX_ACCESS_SIZE
#define BENCHIT_KERNEL_MAX_ACCESS_SIZE (1024*1024)
#endif

#ifndef BENCHIT_KERNEL_ACCESS_STRIDE
#define BENCHIT_KERNEL_ACCESS_STRIDE (2048)
#endif


void init_global_vars(void);

long minlength, maxlength, accessstride;
int rank,size;

void bi_getinfo(bi_info* infostruct){
  int i;
  char buf[200];

  init_global_vars();
  /*infostruct->kernelstring=bi_strdup("Random Memory Access");*/
  infostruct->kerneldescription = bi_strdup("Memory Bandwidth (C)");
  infostruct->codesequence=bi_strdup("for i=1,N#  a[i]+=b[i]*c[i]#");
  infostruct->xaxistext=bi_strdup("Accessed Memory in Byte");
  /*infostruct->kernellanguage=bi_strdup("C");*/
  infostruct->numfunctions=size;
  infostruct->num_measurements=((maxlength-minlength+1)/accessstride);
  
  /* allocating memory for y axis texts and properties */
  allocYAxis(infostruct);
  for (i=0;i<infostruct->numfunctions;i++)
    infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
  infostruct->base_xaxis=2.0;
  infostruct->base_yaxis[0]=0.0;
  for (i=0;i<infostruct->numfunctions;i++){
    sprintf(buf,"Bandwidth (%d active, %d idle)",i+1,size-i-1);
    infostruct->legendtexts[i]=bi_strdup(buf);
  }
  for (i=0;i<infostruct->numfunctions;i++)
    infostruct->yaxistexts[i]=bi_strdup("Byte/s");
#ifdef USE_MPI
  infostruct->kernel_execs_mpi1=1;
#endif
}

void init_global_vars() {
    
  char *envir;
    
#ifdef USE_MPI
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
#else
  rank=0;
  size=1;
#endif
  IDL(3,printf("Init global variables ... "));
  envir=bi_getenv("BENCHIT_KERNEL_MIN_ACCESS_SIZE",1);
  minlength=(envir != 0) ? 1024*atoi(envir) : BENCHIT_KERNEL_MIN_ACCESS_SIZE;
  if(minlength==0) {
    minlength=BENCHIT_KERNEL_MIN_ACCESS_SIZE;
  }
  envir=0;
  envir=bi_getenv("BENCHIT_KERNEL_MAX_ACCESS_SIZE",1);
  maxlength=(envir != 0) ? 1024*atoi(envir) : BENCHIT_KERNEL_MAX_ACCESS_SIZE;
  if(maxlength==0) {
    maxlength=BENCHIT_KERNEL_MIN_ACCESS_SIZE;
  }
  envir=0;
  envir=bi_getenv("BENCHIT_KERNEL_ACCESS_STRIDE",1);
  accessstride=(envir != 0) ? 1024*atoi(envir) : BENCHIT_KERNEL_ACCESS_STRIDE;
  if(accessstride==0) {
    accessstride=BENCHIT_KERNEL_ACCESS_STRIDE;
  }
  IDL(3,printf("done\n"));
}

BI_GET_CALL_OVERHEAD_FUNC((),mem_read(NULL,NULL, NULL, 0))

void *bi_init(int problemSizemax){
  vec_struct *mem;
  long i;
  long maxi;

  IDL(3, printf("Enter init ... "));
  maxi=(minlength+(problemSizemax+1)*accessstride)/sizeof(double);
  mem=malloc(sizeof(vec_struct));
  if (mem==NULL){
    printf("No more core, need %.3f MByte\n", 
	   ((double)minlength*(problemSizemax+1)*accessstride)/(1024*1024));
    exit(127);
  }
  mem->a=malloc(maxi/3*sizeof(double));
  mem->b=malloc(maxi/3*sizeof(double));
  mem->c=malloc(maxi/3*sizeof(double));
  if (mem->a==NULL){
    printf("No more core, need %.3f MByte\n",
           ((double)minlength*(problemSizemax+1)*accessstride)/(1024*1024));
    exit(127);
  }
  if (mem->b==NULL){
    printf("No more core, need %.3f MByte\n",
           ((double)minlength*(problemSizemax+1)*accessstride)/(1024*1024));
    exit(127);
  }
  if (mem->c==NULL){
    printf("No more core, need %.3f MByte\n",
           ((double)minlength*(problemSizemax+1)*accessstride)/(1024*1024));
    exit(127);
  }
  IDL(3, printf("allocated %.3f MByte\n",
		((double)minlength+(problemSizemax+1)*accessstride)/(1024*1024)));
  for (i=0;i<maxi/3;i++) {
    mem->a[i]=(double)(i);
    mem->b[i]=(double)(i);
    mem->c[i]=(double)(i);
  }
  return (void *)(mem);
}

int bi_entry(void *mcb,int problemSize,double *results) {

  static double calloh=0;
  double start, stop;
  int numproc,i,repeats,vectorsize;
  double *a=((vec_struct *)mcb)->a;
  double *b=((vec_struct *)mcb)->b;
  double *c=((vec_struct *)mcb)->c;

  problemSize=(minlength+(problemSize-1)*accessstride); 
  vectorsize=problemSize/sizeof(double)/3;
  repeats=(maxlength/problemSize);
  if(calloh==0) {
    calloh=bi_get_call_overhead();
  }
  
  if (rank==0) results[0]=(double) problemSize;

  for(numproc=1;numproc<=size;numproc++){    

    IDL(2, printf("Enter measurement\n")); 
    mem_read(a, b, c, vectorsize);
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if(rank<numproc){
      start=bi_gettime();
      for (i=0;i<repeats;i++) mem_read(a, b, c, vectorsize);
    }
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    stop=bi_gettime();
    IDL(2, printf("Done\n"));

    if (rank==0){
      results[numproc]=(double)((double)(problemSize+problemSize/3)*numproc/((stop-start-calloh)/repeats-dTimerOverhead));
    }
  }
  return (0);
}

void bi_cleanup(void *mcb){
  double *a=((vec_struct *)mcb)->a;
  double *b=((vec_struct *)mcb)->b;
  double *c=((vec_struct *)mcb)->c;
  free(a);
  free(b);
  free(c);
  free(mcb);
  return;
}
