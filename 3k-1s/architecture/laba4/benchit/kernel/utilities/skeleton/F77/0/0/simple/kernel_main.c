/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/utilities/skeleton/F77/0/0/simple/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: simple Variant of the Fortran-Skeleton
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"

#include "simple.h"

bi_info * myinfo;

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   char * p = 0;
/*
   int ii=0;
   bi_list_t *list;
*/     

   /* get environment variables for the kernel */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
/*
	list = infostruct->list;
   for (ii = 0; ii < infostruct->listsize; ii++){
	   printf("list[%d] = %f\n", ii, list->dnumber);
	   bi_get_list_element(ii);
	   list = list->pnext;
   }
*/   
   infostruct->codesequence = bi_strdup("start kernel; do nothing; ");
   infostruct->kerneldescription = bi_strdup("simple skeleton for c kernels");
   infostruct->xaxistext = bi_strdup("Problem Size");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 1;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   infostruct->yaxistexts[0] = bi_strdup("s");
   infostruct->selected_result[0] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[0] = 1; //logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("time in s");
 
   myinfo = infostruct;
 }



/** Implementation of the bi_init of the BenchIT interface.
 *  Here you have the chance to allocate the memory you need.
 *  It is also possible to allocate the memory at the beginning
 *  of every single measurement and to free the memory thereafter.
 *  But always making use of the same memory is faster.
 *  HAVE A LOOK INTO THE HOWTO !
 */
void* bi_init(int problemSizemax)
{
   return myinfo;
}



/** The central function within each kernel. This function
 *  is called for each measurement step seperately.
 *  @param  mdpv         a pointer to the structure created in bi_init,
 *                       it is the pointer the bi_init returns
 *  @param  problemSize  the actual problemSize
 *  @param  results      a pointer to a field of doubles, the
 *                       size of the field depends on the number
 *                       of functions, there are #functions+1
 *                       doubles
 *  @return 0 if the measurement was sucessfull, something
 *          else in the case of an error
 */
int bi_entry(void * mdpv, int problemSize, double * dresults)
{
  /* timeInSecs: the time for a single measurement in seconds */
  double timeInSecs = 0.0;
  double dprob;
  /* calculate real problemSize */
  
//  printf("bi_get_list_element(%d)\n", iproblemSize);
  dprob = bi_get_list_element(problemSize);
  printf("bi_entry: list[%d] = %f\n", problemSize, dprob);
  //printf("actual problemSize = %d\n", imyproblemSize);

  /* get the actual time
   * do the measurement / your algorythm
   * get the actual time
   */
  problemSize=dprob;
  bi_startTimer();
  fortranfunction_(&problemSize); 
  timeInSecs = bi_stopTimer();

//  fprintf(stderr, "Problemsize=%d, Value=%f\n", imyproblemSize, dres);

  /* store the results in results[1], results[2], ...
  * [1] for the first function, [2] for the second function
  * and so on ...
  * the index 0 always keeps the value for the x axis
  */
  dresults[0] = dprob;
  dresults[1] = timeInSecs;

  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   return;
}
