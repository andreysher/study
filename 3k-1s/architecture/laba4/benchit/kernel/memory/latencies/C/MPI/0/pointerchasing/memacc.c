/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: memacc.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/latencies/C/MPI/0/pointerchasing/memacc.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Memory Access Time (C)
 *******************************************************************/

#include "interface.h"
#include "memacc.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#ifndef BENCHIT_KERNEL_MIN_ACCESS_LENGTH
#define BENCHIT_KERNEL_MIN_ACCESS_LENGTH (2048)
#endif

#ifndef BENCHIT_KERNEL_MAX_ACCESS_LENGTH
#define BENCHIT_KERNEL_MAX_ACCESS_LENGTH (1024*1024)
#endif

#ifndef BENCHIT_KERNEL_ACCESS_STRIDE
#define BENCHIT_KERNEL_ACCESS_STRIDE (2048)
#endif

#ifndef BENCHIT_KERNEL_NUMBER_OF_JUMPS
#define BENCHIT_KERNEL_NUMBER_OF_JUMPS (4000000)
#endif


unsigned int random_number(unsigned int max);
void make_linked_memory(void *mem, int count);
void init_global_vars(void);

long minlength, maxlength, accessstride, numjumps;
int rank,size;

void bi_getinfo(bi_info* infostruct){
  int i;
  char buf[200];

  init_global_vars();
  /*infostruct->kernelstring=bi_strdup("Random Memory Access");*/
  infostruct->kerneldescription = bi_strdup("Memory Access Time (C)");
  infostruct->codesequence=bi_strdup("for i=1,N#  var=memory[random(0..size)]#");
  infostruct->xaxistext=bi_strdup("Accessed Memory in Byte");
  /*infostruct->kernellanguage=bi_strdup("C");*/
  infostruct->numfunctions=size;
  infostruct->num_measurements=((maxlength-minlength+1)/accessstride);
  	/* allocating memory for y axis texts and properties */
  allocYAxis(infostruct);
  
  infostruct->selected_result[0] = SELECT_RESULT_AVERAGE;
  infostruct->base_xaxis=2.0;
  infostruct->base_yaxis[0]=0.0;
  for (i=0;i<infostruct->numfunctions;i++){
    sprintf(buf,"Average Access Time (%d active, %d idle)",i+1,size-i-1);
    infostruct->legendtexts[i]=bi_strdup(buf);
  }
  for (i=0;i<infostruct->numfunctions;i++)
    infostruct->yaxistexts[i]=bi_strdup("s");
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
  envir=bi_getenv("BENCHIT_KERNEL_MIN_ACCESS_LENGTH",1);
  minlength=(envir != 0) ? 1024*atoi(envir) : BENCHIT_KERNEL_MIN_ACCESS_LENGTH;
  if(minlength==0) {
    minlength=BENCHIT_KERNEL_MIN_ACCESS_LENGTH;
  }
  envir=0;
  envir=bi_getenv("BENCHIT_KERNEL_MAX_ACCESS_LENGTH",1);
  maxlength=(envir != 0) ? 1024*atoi(envir) : BENCHIT_KERNEL_MAX_ACCESS_LENGTH;
  if(maxlength==0) {
    maxlength=BENCHIT_KERNEL_MIN_ACCESS_LENGTH;
  }
  envir=0;
  envir=bi_getenv("BENCHIT_KERNEL_ACCESS_STRIDE",1);
  accessstride=(envir != 0) ? 1024*atoi(envir) : BENCHIT_KERNEL_ACCESS_STRIDE;
  if(accessstride==0) {
    accessstride=BENCHIT_KERNEL_ACCESS_STRIDE;
  }
  envir=0;
  envir=bi_getenv("BENCHIT_KERNEL_NUMBER_OF_JUMPS",1);
  numjumps=(envir != 0) ? 1024*atoi(envir) : BENCHIT_KERNEL_NUMBER_OF_JUMPS;
  if(numjumps==0) {
    numjumps=BENCHIT_KERNEL_NUMBER_OF_JUMPS;
  }
  IDL(3,printf("done\n"));
}

BI_GET_CALL_OVERHEAD_FUNC((),jump_around(NULL,0,0));

/** generates a random number between 0 and (max-1)
 *  @param  max maximum random number
 *  @return a random number between 0 and (max-1)
 */
unsigned int random_number(unsigned int max)
{
  return (unsigned int) (((double)max)*rand()/(RAND_MAX+1.0));
}

/** creates a memory are that is randomly linked 
 *  @param mem     the memory area to be used
 *  @param length  the number of bytes that should be used
 */
void make_linked_memory(void *mem, int length) {

  /* some pointers to generate the list */
  void **ptr, **first;
  /** how many ptr we create within the memory */
  int num_ptr=length/sizeof(void *);
  /** the list for all memory locations that are linked */
  int *ptr_numbers;
  /** for the loops */
  int loop_ptrs;
  /** actual random number */
  int act_num;

  /* allocate memory for ptr numbers */
  ptr_numbers=(int *) malloc(num_ptr*sizeof(int));
  if(num_ptr>0 && ptr_numbers==NULL)
    {
      printf("no more core in make_linked_mem()\n");
      bi_cleanup(mem);
      exit(1);
    }
  /* initialize ptr numbers, the 0 is used as the first
   * number
   */
  for(loop_ptrs=1; loop_ptrs<num_ptr; loop_ptrs++)
    ptr_numbers[loop_ptrs-1]=loop_ptrs;

  /* init first ptr with first memory location */
  ptr=(void **)mem;
  first=ptr;
   
  num_ptr--;

  while(num_ptr>1) {
    /* get a random position within the
       remaining list */
    act_num=random_number(num_ptr);
    /* create a link from the last ptr 
       to this ptr */
    *ptr=(void *) (first+ptr_numbers[act_num]);
    /* move pointer to new memory location */
    ptr=first+ptr_numbers[act_num];
    /* remove used ptr number from list of
       pointer numbers, just copies the last 
       number to the actual position */
    ptr_numbers[act_num]=ptr_numbers[num_ptr-1];
    num_ptr--;
  }

  /* the last number is linked to the first */
  *ptr=(void *) first;

  /* free the ptr list */
  free(ptr_numbers);
  IDL(4,printf("\n"));
}

void *bi_init(int problemSizemax){
  void *mem;

  IDL(3, printf("Enter init ... "));
  mem=malloc(minlength+(problemSizemax+1)*accessstride);
  if (mem==NULL){
    printf("No more core, need %.3f MByte\n", 
	   ((double)minlength*(problemSizemax+1)*accessstride)/(1024*1024));
    exit(127);
  }
  IDL(3, printf("allocated %.3f MByte\n",
		((double)minlength+(problemSizemax+1)*accessstride)/(1024*1024)));
  return (mem);
}

int bi_entry(void *mcb,int problemSize,double *results) {

  static double calloh=0;
  double start, stop;
  int numproc;

  problemSize=minlength+(problemSize-1)*accessstride; 
  if(calloh==0) {
    calloh=bi_get_call_overhead();
    if(calloh==0)
      calloh=dTimerGranularity;
  }

  if (rank==0) results[0]=(double) problemSize;

  for(numproc=1;numproc<=size;numproc++){    

    if(rank<numproc)
    {
      IDL(2, printf("Making structure\n"));
      make_linked_memory(mcb, problemSize);
      IDL(2, printf("Enter measurement\n"));
      jump_around(mcb, problemSize,numjumps); 
      start=bi_gettime();
      jump_around(mcb, problemSize,numjumps);
      stop=bi_gettime();
      IDL(2, printf("Done\n"));
    }

    if (rank==0){
      /* we have always numjumps memory accesses */
      results[numproc]=(double)((stop-start-calloh-dTimerOverhead)/((double)numjumps));
    }
  }
  return (0);
}

void bi_cleanup(void *mcb){
  free(mcb);
  return;
}
