/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/applications/reflection/C/0/0/soccergoal/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare the possibility to hit the goal when the pole have a squared or round form
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"
#include "soccer.h"

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   char *p = 0;
   

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

   infostruct->codesequence = bi_strdup("start kernel; count goals for squared pole; count goals for round pole;");
   infostruct->kerneldescription = bi_strdup("compare the possibility to hit the goal when the pole have a squared or round form");
   infostruct->xaxistext = bi_strdup("n shot points, n pole-hit points");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 4;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   infostruct->yaxistexts[0] = bi_strdup("s");
   infostruct->selected_result[0] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[0] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("time in s for square pole");

   infostruct->yaxistexts[1] = bi_strdup("s");
   infostruct->selected_result[1] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[1] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[1] = bi_strdup("time in s for round pole");
 
   infostruct->yaxistexts[2] = bi_strdup("goals");
   infostruct->selected_result[2] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[2] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[2] = bi_strdup("goals for square pole");
 
   infostruct->yaxistexts[3] = bi_strdup("goals");
   infostruct->selected_result[3] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[3] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[3] = bi_strdup("goals for round pole");
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
   return (void *)NULL;
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
  /* dstart, dend: the start and end time of the measurement */
  /* dtime: the time for a single measurement in seconds */
  double dstart[2], dend[2], dtime[2];
  /* flops stores the calculated FLOPS */
  double dres[2];
  /* ii is used for loop iterations */
  myinttype ii = 0, imyproblemSize = 0;

  /* calculate real problemSize */
  imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);

  /* check wether the pointer to store the results in is valid or not */
  if (dresults == NULL) return 1;

  /* get the actual time
   * do the measurement / your algorythm
   * get the actual time
   */
  for(ii=0; ii<2; ii++){
    dstart[ii] = bi_gettime(); 
    dres[ii] = soccergoal(&imyproblemSize, &ii);
    dend[ii] = bi_gettime();
  }

  /* calculate the used time and FLOPS */
  for(ii=0; ii<2; ii++){
    dtime[ii] = dend[ii] - dstart[ii];
    dtime[ii] -= dTimerOverhead;
  }  

  /* If the operation was too fast to be measured by the timer function,
   * mark the result as invalid 
   */
  for(ii=0; ii<2; ii++){
    if(dtime[ii] < dTimerGranularity) dtime[ii] = INVALID_MEASUREMENT;
  }  

  /* store the results in results[1], results[2], ...
  * [1] for the first function, [2] for the second function
  * and so on ...
  * the index 0 always keeps the value for the x axis
  */
  dresults[0] = (double)imyproblemSize;
  dresults[1] = dtime[0];
  dresults[2] = dtime[1];
  dresults[3] = dres[0];
  dresults[4] = dres[1];

  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   return;
}
