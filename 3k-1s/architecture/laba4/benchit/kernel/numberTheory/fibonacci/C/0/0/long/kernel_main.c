/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numberTheory/fibonacci/C/0/0/long/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for calc of Fibonacci number (iterative / recursive)
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "work.h"

#define NUM_FUNC 2

/* Reads the environment variables used by this kernel. 
 * see interface.h for bi_getenv("name", exit_on_error)
 */
void evaluate_environment(mydata_t * pmydata) {
   char * p = 0;

   /* add additional parameters, except BENCHIT_KERNEL_PROBLEMLIST from the parameters file
    * BENCHIT_KERNEL_PROBLEMLIST will be done in bi_getinfo */
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   char * p = 0;
   mydata_t * penv;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence
      = bi_strdup("iterative calculation; recursive calculation; ");
   infostruct->kerneldescription
      = bi_strdup("Execution time of Fibonacci number generation (iterative / recursive)");
   infostruct->xaxistext = bi_strdup("number");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = NUM_FUNC;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   infostruct->yaxistexts[0] = bi_strdup("s");
   infostruct->selected_result[0] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[0] = 100; //logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("iterative");

   /* setting up y axis texts and properties */
   infostruct->yaxistexts[1] = bi_strdup("s");
   infostruct->selected_result[1] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[1] = 100; //logarythmic axis 10^x
   infostruct->legendtexts[1] = bi_strdup("recursive");

   /* free all used space */
   if (penv)
      free(penv);
}

/** Implementation of the bi_init of the BenchIT interface.
 *  Here you have the chance to allocate the memory you need.
 *  It is also possible to allocate the memory at the beginning
 *  of every single measurement and to free the memory thereafter.
 *  But always making use of the same memory is faster.
 *  HAVE A LOOK INTO THE HOWTO !
 */
void* bi_init(int problemSizemax) {
   mydata_t * pmydata;

   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
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
int bi_entry(void * mdpv, int iproblemSize, double * dresults) {
   /* dstart, dend: the start and end time of the measurement */
   /* dtime: the time for a single measurement in seconds */
   double dstart[2], dend[2], dtime[2];
   /* count stores the #Operations */
   double count = 0.0;
   /* ii is used for loop iterations */
   myinttype ii = 0, imyproblemSize, numloop[2];
   /* cast void* pointer */
   mydata_t * pmydata = (mydata_t *) mdpv;

   /* get current problem size from problemlist */
   imyproblemSize = (long)bi_get_list_element(iproblemSize);

   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL)
      return 1;

   dresults[0] = (double)imyproblemSize;

   /* get the actual time
    * do the measurement / your algorythm
    * get the actual time
    */
   numloop[0] = 1;
   do {
      dstart[0] = bi_gettime();
      for(ii=0; ii<numloop[0]; ii++) (void)linfib(imyproblemSize);
      dend[0] = bi_gettime();

      numloop[0] <<= 3; /* numloop = numloop * 8 */
   } while (dend[0]-dstart[0] < 0.01);
   numloop[0] >>= 3;    /* undo last shift */

   numloop[1] = 1;
   do {
      dstart[1] = bi_gettime();
      for(ii=0; ii<numloop[1]; ii++) (void)recfib(imyproblemSize);
      dend[1] = bi_gettime();

      numloop[1] <<= 3; /* numloop = numloop * 8 */
   } while (dend[1]-dstart[1] < 0.01);
   numloop[1] >>= 3;    /* undo last shift */

   for(ii=0; ii<NUM_FUNC; ii++) {
      /* calculate the used time */
      dtime[ii] = dend[ii] - dstart[ii];
      dtime[ii] -= dTimerOverhead;

      /* If the operation was too fast to be measured by the timer function,
       * mark the result as invalid 
       */
      if (dtime[ii] < dTimerGranularity)
         dtime[ii] = INVALID_MEASUREMENT;

      /* store the results in results[1], results[2], ...
       * [1] for the first function, [2] for the second function
       * and so on ...
       * the index 0 always keeps the value for the x axis
       */
      dresults[ii+1] = (dtime[ii]!=INVALID_MEASUREMENT) ? dtime[ii] / numloop[ii]
         : INVALID_MEASUREMENT;
   }

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv) {
   mydata_t * pmydata = (mydata_t*)mdpv;

   if (pmydata)
      free(pmydata);

   return;
}

