/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/elementary/C/0/0/sqrt/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for Square Root addict to input value
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "work.h"

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
      = bi_strdup("for(a=0; a<1000; a++) do the mathematical operation; ");
   infostruct->kerneldescription
      = bi_strdup("Square Root addict to input value");
   infostruct->xaxistext = bi_strdup("Argument");
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
   infostruct->yaxistexts[0] = bi_strdup("Op/s");
   infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[0] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("sqrt");

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

/** stuff from old kernel
 */
double count_(int *version, int *size) {
   switch (*version) {
      default:
         return 1000/* *1000 */;
   }
}

/** stuff from old kernel
 */
void useversion_(int *version) {
   switch (*version) {
      case 0:
         mathop=mathopsqrt_;
         break;
      default:
         printf("Kernel Error: Illegal Version\n");
         exit(127);
   }
}

/** stuff from old kernel
 *  determine call overhead (function call mathopsin_ / mathopcos_ and if statement)
 */
double getseqentryoverhead() {
   double start, stop;
   int nu=0, s;

   start = bi_gettime();
   for(s=0; s<1000; s++) {
      entry_(&nu);
   }
   stop = bi_gettime();
   return (stop - start) / 1000;
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
   double dstart = 0.0, dend = 0.0, dtime = 0.0, calloverhead = 0.0;
   /* count stores the #Operations */
   double count = 0.0;
   /* stuff from old kernel */
   int v = 0;
   /* ii is used for loop iterations */
   myinttype ii = 0, imyproblemSize;
   /* cast void* pointer */
   mydata_t * pmydata = (mydata_t *) mdpv;

   /* get current problem size from problemlist */
   imyproblemSize = (myinttype)bi_get_list_element(iproblemSize);

   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL)
      return 1;

   useversion_(&v);
   calloverhead = getseqentryoverhead();

   /* get the actual time
    * do the measurement / your algorythm
    * get the actual time
    */
   dstart = bi_gettime();
   entry_(&imyproblemSize);
   dend = bi_gettime();

   //  fprintf(stderr, "Problemsize=%d, Value=%f\n", imyproblemSize, dres);

   /* calculate the used time and #Operations */
   dtime = dend - dstart;
   dtime -= dTimerOverhead;
   dtime -= calloverhead;
   count = count_(&v, &imyproblemSize);

   /* If the operation was too fast to be measured by the timer function,
    * mark the result as invalid 
    */
   if (dtime < dTimerGranularity)
      dtime = INVALID_MEASUREMENT;

   /* store the results in results[1], results[2], ...
    * [1] for the first function, [2] for the second function
    * and so on ...
    * the index 0 always keeps the value for the x axis
    */
   dresults[0] = (double)imyproblemSize;
   dresults[1] = (dtime!=INVALID_MEASUREMENT) ? count / dtime
      : INVALID_MEASUREMENT;

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

