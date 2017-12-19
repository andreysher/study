/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/cacheasso/C/0/0/pointerchasing/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Kernel to determine the association of the L1 data cache
 *******************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"

#include "cache_jump.h"

/* Reads the environment variables used by this kernel. */
void evaluate_environment(mydata_t * pmydata) {
   char * p = 0;
   
   p = bi_getenv("BENCHIT_KERNEL_MINSTRIDE", 0);
   if (p == NULL) pmydata->minstride = MINSTRIDE_DEFAULT;
   else pmydata->minstride = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_MAXSTRIDE", 0);
   if (p == NULL) pmydata->maxstride = MAXSTRIDE_DEFAULT;
   else pmydata->maxstride = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_MAXLOOPLENGTH", 0);
   if (p == NULL) pmydata->maxlooplength = MAXLOOPLENGTH_DEFAULT;
   else pmydata->maxlooplength = atoi(p);
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   mydata_t * penv;
   myinttype ii, k;
   char buff[80];
   
   penv = (mydata_t *) malloc(sizeof(mydata_t));
   
   /* get environment variables for the kernel */
   evaluate_environment(penv);
   infostruct->codesequence = bi_strdup("ptr1->ptr2->ptr3->...->ptrN->ptr1");
   infostruct->kerneldescription = bi_strdup("Kernel to determine the association of the L1 data cache");
   infostruct->xaxistext = bi_strdup("access loop length");
   infostruct->num_measurements = penv->maxlooplength;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = (int) (log(penv->maxstride)/log(2.0)-log(penv->minstride)/log(2.0))+1;
   
   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   k = penv->minstride;
   for (ii=0; ii<infostruct->numfunctions; ii++) {
      infostruct->yaxistexts[ii] = bi_strdup("s");
      infostruct->selected_result[ii] = SELECT_RESULT_LOWEST;
      infostruct->base_yaxis[ii] = 0; //logarythmic axis 10^x
      sprintf(buff, "stride %d K", k/1024);
      infostruct->legendtexts[ii]=bi_strdup(buff);
      k *= 2;
   }
   
   /* free all used space */
   if (penv) free(penv);
}

/** stuff from old kernel
 */
double getseqentryoverhead() {
   double start, stop;
   int s;
   
   start = bi_gettime();
   for (s=0; s<1000; s++)
      jump_around(NULL, 0);
   stop = bi_gettime();
   return (stop-start) / 1000;
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
   
   if (problemSizemax > pmydata->maxlooplength) {
      printf("Illegal maximum problem size\n");
      exit(127);
   }
   
   pmydata->mem = (void*)malloc((pmydata->maxstride+1)*(pmydata->maxlooplength+1)*sizeof(void*));
   if (pmydata->mem==NULL) {
      printf("No more core, need %.3f MByte\n", ((double) (sizeof(void*)*(pmydata->maxstride+1)*(pmydata->maxlooplength+1)))/(1024*1024));
      exit(127);
   }
   IDL(3, printf("allocated %.3f MByte\n", ((double) (sizeof(void*)*(pmydata->maxstride+1)*(pmydata->maxlooplength+1)))/(1024*1024)))
   
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
   double dstart = 0.0, dend = 0.0, dtime = 0.0;
   static double calloverhead = 0.0;
   /* ii is used for loop iterations */
   myinttype loop_var, loop, minstride, maxstride;
   /* cast void* pointer */
   mydata_t * pmydata = (mydata_t *) mdpv;
   
   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL) return 1;
   
   if (calloverhead==0) {
      calloverhead = getseqentryoverhead();
      if (calloverhead==0)
         calloverhead = MINTIME;
   }
   
   minstride = pmydata->minstride;
   maxstride = pmydata->maxstride;
   
   /* set value for x-axis */
   dresults[0] = (double)iproblemSize;
   
   /* do the measurement */
	for (loop_var=minstride, loop=1; loop_var<=maxstride; loop_var*=2, loop++) {
		IDL(2, printf("Making structure\n"));
	 	make_jump_structure(pmydata->mem, iproblemSize, loop_var);
      
      /* get the actual time
       * do the measurement / your algorythm
       * get the actual time
       */
		IDL(2, printf("Enter measurement\n"));
      dstart = bi_gettime(); 
      jump_around(pmydata->mem, iproblemSize);
      dend = bi_gettime();
		IDL(2, printf("Done\n"));
      
      /* calculate the used time */
      dtime = dend - dstart;
      dtime -= dTimerOverhead;
      dtime -= calloverhead;
      
      /* If the operation was too fast to be measured by the timer function,
       * mark the result as invalid 
       */
      if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;
      
      /* store the results in results[1], results[2], ...
       * [1] for the first function, [2] for the second function
       * and so on ...
       * the index 0 always keeps the value for the x axis
       */
		dresults[loop] = (dtime!=INVALID_MEASUREMENT) ? dtime / (double)4000000.0
         : INVALID_MEASUREMENT;
	}
   
   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv) {
   mydata_t * pmydata = (mydata_t*)mdpv;
   
   if (pmydata) {
      if (pmydata->mem)
         free(pmydata->mem);
      free(pmydata);
   }
   return;
}

