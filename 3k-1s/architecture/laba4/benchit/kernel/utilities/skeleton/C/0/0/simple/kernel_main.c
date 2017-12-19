/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/utilities/skeleton/C/0/0/simple/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: simple Variant of the c-Skeleton
 *******************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"

#include "simple.h"


/* Reads the environment variables used by this kernel. */
void evaluate_environment(mydata_t * pmydata)
{
   int errors = 0;
   char * p = 0;
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMSIZE_MIN", 0);
   if (p == NULL) errors++;
   else pmydata->min = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMSIZE_MAX", 0);
   if (p == NULL) errors++;
   else pmydata->max = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMSIZE_INCREMENT", 0);
   if (p == NULL) errors++;
   else pmydata->increment = atoi(p);
   if (errors > 0)
   {
      fprintf(stderr, "There's at least one environment variable not set!\n");
      exit(1);
   }
   pmydata->steps = (myinttype) (pmydata->max - pmydata->min + 1) / pmydata->increment;
   if ((pmydata->max - pmydata->min + 1) % pmydata->increment != 0) pmydata->steps++;
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   mydata_t * penv;
   
   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   evaluate_environment(penv);
   infostruct->codesequence = bi_strdup("start kernel; do nothing; ");
   infostruct->kerneldescription = bi_strdup("simple skeleton for c kernels");
   infostruct->xaxistext = bi_strdup("Problem Size");
   infostruct->num_measurements = penv->steps;
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
      infostruct->base_yaxis[0] = 10; //logarythmic axis 10^x
      infostruct->legendtexts[0] = bi_strdup("time in s");
 
   /* free all used space */
   if (penv) free(penv);
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
   mydata_t * pmydata;

   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);
//  fprintf(stderr, "max=%d, min=%d, increment=%d, steps=%d\n",pmydata->max, pmydata->min, pmydata->increment, pmydata->steps);
   return (void *)pmydata;
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
int bi_entry(void * mdpv, int iproblemSize, double * dresults)
{
  /* flops stores the calculated FLOPS */
  double dres = 0.0;
  /* ii is used for loop iterations */
  myinttype ii = 0, imyproblemSize = (myinttype) iproblemSize;
  /* cast void* pointer */
  mydata_t * pmydata = (mydata_t *) mdpv;

  /* calculate real problemSize */
  imyproblemSize = pmydata->min + ((imyproblemSize - 1)  * pmydata->increment);

  /* check wether the pointer to store the results in is valid or not */
  if (dresults == NULL) return 1;

  /* get the actual time
   * do the measurement / your algorythm
   * get the actual time
   */
  bi_startTimer();
  dres = simple(&imyproblemSize); 
  double timeInSecs=bi_stopTimer();

//  fprintf(stderr, "Problemsize=%d, Value=%f\n", imyproblemSize, dres);

  /* store the results in results[1], results[2], ...
  * [1] for the first function, [2] for the second function
  * and so on ...
  * the index 0 always keeps the value for the x axis
  */
  dresults[0] = (double)imyproblemSize;
  dresults[1] = timeInSecs;

  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   mydata_t * pmydata = (mydata_t*)mdpv;
   if (pmydata) free(pmydata);
   return;
}
