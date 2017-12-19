/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/applications/MGV/F95/OpenMP/0/recursive/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: multigrid methode for 2D Poisson equation
 *******************************************************************/

#include <omp.h>
#include "2dPoisson.h"

#define NUM_FUNC 2

/* Reads the environment variables used by this kernel. */
void evaluate_environment(mydata_t * pmydata)
{
   int errors = 0;
   char * p = 0;

   p = bi_getenv("BENCHIT_KERNEL_OUTPUT", 0);
   if (p == NULL) errors++;
   else pmydata->output = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_OUTPUTFORM", 0);
   if (p == NULL) errors++;
   else pmydata->outputform = atoi(p);
   
   if (errors > 0)
   {
      fprintf(stderr, "There's at least one environment variable not set!\n");
      exit(1);
   }
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   char *p = 0;
   myinttype ii;
   mydata_t * penv;
   
   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence = bi_strdup("generate matries, perform the V-cycles, restriction, prolongation, ...");
   infostruct->kerneldescription = bi_strdup("MGV");
   infostruct->xaxistext = bi_strdup("level => (2^level+1)^2 x (2^level+1)^2 - Matrix");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 1;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = NUM_FUNC;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   if(penv->output) {
      infostruct->yaxistexts[ii] = bi_strdup("s");
      infostruct->selected_result[ii] = SELECT_RESULT_LOWEST;
      infostruct->base_yaxis[ii] = 0; //logarythmic axis 10^x
      infostruct->legendtexts[ii] = bi_strdup("time in s");
      ii++;

      infostruct->yaxistexts[ii] = bi_strdup("residual quotient");
      infostruct->selected_result[ii] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[ii] = 0; //logarythmic axis 10^x
      infostruct->legendtexts[ii] = bi_strdup("residual quotient");
   } else { 
      infostruct->yaxistexts[ii] = bi_strdup("FLOPS");
      infostruct->selected_result[ii] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[ii] = 0; //logarythmic axis 10^x
      infostruct->legendtexts[ii] = bi_strdup("FLOPS (ijk)");
      ii++;

      infostruct->yaxistexts[ii] = bi_strdup("residual quotient");
      infostruct->selected_result[ii] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[ii] = 0; //logarythmic axis 10^x
      infostruct->legendtexts[ii] = bi_strdup("residual quotient");
   }

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
   myinttype maxsize;
   myinttype numthreads = 0;

   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);
   
   numthreads = omp_get_max_threads();
   IDL(0, printf(" [ Kernel uses %d OpenMP Threads ] ", numthreads));
  
   maxsize = (myinttype) bi_get_list_maxelement();
   pmydata->maxsize = maxsize;

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
  /* dtime: the time for a single measurement in seconds */
  double dtime = 0.0;
  /* flops stores the calculated FLOPS */
  double dres = 0.0;
  
  double w = 0.0, L1 = 0.0, L2 = 0.0, omega = 0.0;

  myinttype level = 0;
  myinttype maxlevel = 0, outputform = 0, v1 = 0, v2 = 0;
  
  /* cast void* pointer */
  mydata_t * pmydata = (mydata_t *) mdpv;

  /* calculate real problemSize */
  level = (myinttype) bi_get_list_element(iproblemSize);
  IDL(3, printf("\nlevel=%d",level));
  
  /* set additional parameters */
  maxlevel = pmydata->maxsize;
  outputform = pmydata->outputform;
  v1 = 5;
  v2 = 5;
  w = 0.5;
  L1 = 1.0;
  L2 = 1.0;
  
  /* check wether the pointer to store the results in is valid or not */
  if (dresults == NULL) return 1;

  /* get the actual time
   * do the measurement / your algorythm
   * get the actual time
   */
  fortran_entry_(&level, &maxlevel, &outputform, &v1, &v2, &w, &L1, &L2, &dtime, &omega, &dres);

  /* If the operation was too fast to be measured by the timer function,
   * mark the result as invalid 
   */
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;

  /* store the results in results[1], results[2], ...
  * [1] for the first function, [2] for the second function
  * and so on ...
  * the index 0 always keeps the value for the x axis
  */
  dresults[0] = (double)level;
  if(pmydata->output) {
    dresults[1] = dtime;
    dresults[2] = omega;
  } else {
    dresults[1]
       = (dtime != INVALID_MEASUREMENT) ? dres / dtime : INVALID_MEASUREMENT;
    dresults[2] = omega;
  }

  return 0;
}

/** To call from Fortran the fflush function
 */
void f90fflush_(){ fflush(stdout); fflush(stderr); }

/** Called from Fortran to get a good timer (small granularity)
 */
void bi_gettime_(double * time) {
  *time = bi_gettime();
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   mydata_t * pmydata = (mydata_t*)mdpv;
   if (pmydata) free(pmydata);
   return;
}


