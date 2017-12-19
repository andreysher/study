/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/string/toUpperCase/C/SSE2_Intrinsics/0/char/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: SSE String Operations
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mm_malloc.h>
#include "interface.h"

#include "simple.h"


/* Reads the environment variables used by this kernel. */
void evaluate_environment(mydata_t * pmydata)
{
   int errors = 0;
   char * p = 0;

   p = bi_getenv("BENCHIT_KERNEL_PROBLEMSIZE_OFFSET", 0);
   if (p == NULL) errors++;
   else pmydata->offset = atoi(p);
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
   mydata_t * penv;
   int i;
   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence = bi_strdup("toUppercase(char* string, int length)");
   infostruct->kerneldescription = bi_strdup("SSE to upper case");
   infostruct->xaxistext = bi_strdup("String length");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 1;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 3;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
      infostruct->yaxistexts[i] = bi_strdup("s");
      infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
      infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   }
      infostruct->legendtexts[0] = bi_strdup("standard");
      infostruct->legendtexts[1] = bi_strdup("SSE unaligned");
      infostruct->legendtexts[2] = bi_strdup("SSE aligned");
 
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

   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);

   maxsize = (myinttype) bi_get_list_maxelement();
   pmydata->maxsize = maxsize;

   pmydata->field = (char*)_mm_malloc(maxsize*sizeof(char)+pmydata->offset, 64);

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
  /* dstart, dend: the start and end time of the measurement */
  /* dtime: the time for a single measurement in seconds */
  double dstart = 0.0, dend = 0.0, dtime = 0.0;
  /* flops stores the calculated FLOPS */
  double dres = 0.0;
  /* ii is used for loop iterations */
  myinttype ii = 0, imyproblemSize = 0;
  /* cast void* pointer */
  mydata_t * pmydata = (mydata_t *) mdpv;
  myinttype offset = pmydata->offset;
  /* calculate real problemSize */
  imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);
  
	for (ii=0;ii<imyproblemSize+offset;ii++)
	{
		if (ii%2==0)
			pmydata->field[ii]='a';
		else
			pmydata->field[ii]='B';
	}
  /* check wether the pointer to store the results in is valid or not */
  if (dresults == NULL) return 1;
	
  /* get the actual time
   * do the measurement / your algorythm
   * get the actual time
   */
  dstart = bi_gettime(); 
  dres = toUpperCase(&(pmydata->field[offset]), imyproblemSize); 
  dend = bi_gettime();

//  fprintf(stderr, "Problemsize=%d, Value=%f\n", imyproblemSize, dres);

  /* calculate the used time and FLOPS */
  dtime = dend - dstart;
  dtime -= dTimerOverhead;
      
  /* If the operation was too fast to be measured by the timer function,
   * mark the result as invalid 
   */
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;

  /* store the results in results[1], results[2], ...
  * [1] for the first function, [2] for the second function
  * and so on ...
  * the index 0 always keeps the value for the x axis
  */
  dresults[0] = (double)imyproblemSize;
  dresults[1] = dtime;

	for (ii=0;ii<imyproblemSize+offset;ii++)
	{
		if (ii%2==0)
			pmydata->field[ii]='a';
		else
			pmydata->field[ii]='B';
	}
  /* check wether the pointer to store the results in is valid or not */
  if (dresults == NULL) return 1;
	
  /* get the actual time
   * do the measurement / your algorythm
   * get the actual time
   */
  dstart = bi_gettime(); 
  dres = toUpperCaseSSEunaligned(&(pmydata->field[offset]), imyproblemSize); 
  dend = bi_gettime();

//  fprintf(stderr, "Problemsize=%d, Value=%f\n", imyproblemSize, dres);

  /* calculate the used time and FLOPS */
  dtime = dend - dstart;
  dtime -= dTimerOverhead;
      
  /* If the operation was too fast to be measured by the timer function,
   * mark the result as invalid 
   */
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;

  /* store the results in results[1], results[2], ...
  * [1] for the first function, [2] for the second function
  * and so on ...
  * the index 0 always keeps the value for the x axis
  */
  dresults[2] = dtime;

	for (ii=0;ii<imyproblemSize+offset;ii++)
	{
		if (ii%2==0)
			pmydata->field[ii]='a';
		else
			pmydata->field[ii]='B';
	}
  /* check wether the pointer to store the results in is valid or not */
  if (dresults == NULL) return 1;
	
  /* get the actual time
   * do the measurement / your algorythm
   * get the actual time
   */
  dstart = bi_gettime(); 
  dres = toUpperCaseSSE(pmydata->field, imyproblemSize); 
  dend = bi_gettime();

//  fprintf(stderr, "Problemsize=%d, Value=%f\n", imyproblemSize, dres);

  /* calculate the used time and FLOPS */
  dtime = dend - dstart;
  dtime -= dTimerOverhead;
      
  /* If the operation was too fast to be measured by the timer function,
   * mark the result as invalid 
   */
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;

  /* store the results in results[1], results[2], ...
  * [1] for the first function, [2] for the second function
  * and so on ...
  * the index 0 always keeps the value for the x axis
  */
  dresults[3] = dtime;
  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   mydata_t * pmydata = (mydata_t*)mdpv;
   if (pmydata)
   {
    if (pmydata->field)
   	 _mm_free(pmydata->field);
   	free(pmydata);
   }
   return;
}

