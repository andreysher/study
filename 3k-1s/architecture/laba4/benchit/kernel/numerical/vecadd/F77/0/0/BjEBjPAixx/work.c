/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/vecadd/F77/0/0/BjEBjPAixx/work.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "work.h"

#define REPETITIONS_DEFAULT 1

/* Reads the environment variables used by this kernel. 
 * see interface.h for bi_getenv("name", exit_on_error)
 */
void evaluate_environment(mydata_t * pmydata) {
   char *p = 0;

   /* add additional parameters, except BENCHIT_KERNEL_PROBLEMLIST from the
    * parameters file BENCHIT_KERNEL_PROBLEMLIST will be done in bi_getinfo */
   p = bi_getenv("BENCHIT_KERNEL_REPETITIONS", 0);
   if (p == NULL)
      pmydata->repetitions = REPETITIONS_DEFAULT;
   else
      pmydata->repetitions = atoi(p);
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   int ii;
   char buffer[200];
   char *p = 0;
   mydata_t *penv;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence = bi_strdup("vecadd ixxj: b(j)=b(j)+a(ixx)");
   infostruct->kerneldescription =
      bi_strdup("Measure flop/s of vector addition: b(j)=b(j)+a(ixx)");
   infostruct->xaxistext = bi_strdup("vector size");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 8;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   for (ii = 0; ii < infostruct->numfunctions; ii++) {
      infostruct->yaxistexts[ii] = bi_strdup("flop/s");
      infostruct->selected_result[ii] = SELECT_RESULT_HIGHEST;
      infostruct->log_yaxis[ii] = 0;
      infostruct->base_yaxis[ii] = 0;
      sprintf(buffer, "unrolled %d", ii + 1);
      infostruct->legendtexts[ii] = bi_strdup(buffer);
   }

   if (DEBUGLEVEL > 3) {
      for (ii = 0; ii < infostruct->numfunctions; ii++) {
         printf("yaxis[%2d]=%s\t\t\tlegend[%2d]=%s\n", ii,
                infostruct->yaxistexts[ii], ii, infostruct->legendtexts[ii]);
      }
   }
}

/** Implementation of the bi_init of the BenchIT interface.
 *  Here you have the chance to allocate the memory you need.
 *  It is also possible to allocate the memory at the beginning
 *  of every single measurement and to free the memory thereafter.
 *  But always making use of the same memory is faster.
 *  HAVE A LOOK INTO THE HOWTO !
 */
void *bi_init(int problemSizemax) {
   mydata_t *pmydata;

   pmydata = (mydata_t *) malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);

   return (void *)pmydata;
}

/* This function allocates the memory for the vectors and 
 * and assings the numbers.
 ***/
void allocateANDtouch(mydata_t * pmem, int *pisize) {
   double *pda, *pdb;
   int ii, maxProblemSize;

   pmem->pda = (double *)malloc((*pisize + 1) * sizeof(double));
   pmem->pdb = (double *)malloc((*pisize + 1) * sizeof(double));

   if ((pmem->pda == NULL) || (pmem->pdb == NULL)) {
      printf("malloc (%.2f MB) failed in bi_init()\n",
             (double)(2 * (*pisize) * sizeof(double))
             / (double)(1024 * 1024));
      bi_cleanup(pmem);
      exit(127);
   }

   pda = pmem->pda;
   pdb = pmem->pdb;

   for (ii = *pisize - 1; ii >= 0; ii--) {
      pda[ii] = (double)rand();
      pdb[ii] = (double)rand();
/*
      pda[ii]= 1;
      pdb[ii]= 1;
*/
   }
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
int bi_entry(void *mdpv, int iproblemSize, double *dresults) {
   /* dstart, dend: the start and end time of the measurement */
   /* dtime: the time for a single measurement in seconds */
   double dstart = 0.0, dend = 0.0, dtime[8], flop = 0.0;
   /* ii, ij is used for loop iterations */
   myinttype ii = 0, imyproblemSize = 0, iunrolled = 0;
   /* cast void* pointer */
   mydata_t *pmydata = (mydata_t *) mdpv;

   IDL(1, printf("reached function bi_entry\n"));

   /* get current problem size from problemlist */
   imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);

   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL)
      return 1;

   /* get the actual time do the measurement / your algorythm get the actual
    * time */
   for (iunrolled = 1; iunrolled < 9; iunrolled++) {
      allocateANDtouch(pmydata, &imyproblemSize);

      dstart = bi_gettime();
      vecadd_(&imyproblemSize, &pmydata->repetitions, &iunrolled, pmydata->pda,
              pmydata->pdb);
      dend = bi_gettime();

      /* calculate the used time and #Operations */
      dtime[iunrolled - 1] = dend - dstart;
      dtime[iunrolled - 1] -= dTimerOverhead;

      free(pmydata->pda);
      free(pmydata->pdb);
      pmydata->pda = NULL;
      pmydata->pdb = NULL;
   }

   flop = (double)imyproblemSize * (double)imyproblemSize *
          (double)pmydata->repetitions;

   dresults[0] = (double)imyproblemSize;
   for (ii = 0; ii < 8; ii++) {
      /* If the operation was too fast to be measured by the timer function,
       * mark the result as invalid */
      if (dtime[ii] < dTimerGranularity)
         dtime[ii] = INVALID_MEASUREMENT;

      /* store the results in results[1], results[2], ... [1] for the first
       * function, [2] for the second function and so on ... the index 0 always
       * keeps the value for the x axis */
      dresults[ii + 1] =
         (dtime[ii] !=
          INVALID_MEASUREMENT) ? flop / dtime[ii] : INVALID_MEASUREMENT;
   }

   IDL(1, printf("completed function bi_entry\n"));

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void *mdpv) {
   mydata_t *pmydata = (mydata_t *) mdpv;

   if (pmydata) {
      /* free matricies */
      if (pmydata->pda)
         free(pmydata->pda);
      if (pmydata->pdb)
         free(pmydata->pdb);

      free(pmydata);
   }
   return;
}

