/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 11 2009-09-18 15:25:01Z domke $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/C/PThread/0/1_malloc/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension, for a given number of posix threads,
 *         one malloc for biggest dimension
 *******************************************************************/

#include "jacobi.h"

#define RUNS 2                         /* variations (ij and ji = 2) */
#define NUM_FUNC 2                     /* flop/s and time */

#define PTHREADS_COUNT_DEFAULT 1
#define MITS_DEFAULT 500

/* Reads the environment variables used by this kernel. 
 * see interface.h for bi_getenv("name", exit_on_error)
 */
void evaluate_environment(mydata_t * pmydata) {
   char *p = 0;

   /* add additional parameters, except BENCHIT_KERNEL_PROBLEMLIST from the
    * parameters file BENCHIT_KERNEL_PROBLEMLIST will be done in bi_getinfo */
   p = bi_getenv("BENCHIT_KERNEL_PTHREAD_COUNT", 0);
   if (p == NULL)
      pmydata->threadCount = PTHREADS_COUNT_DEFAULT;
   else
      pmydata->threadCount = atoi(p);
   /* use squared number of threads */
   pmydata->threadCount = pmydata->threadCount * pmydata->threadCount;

   p = bi_getenv("BENCHIT_KERNEL_JACOBI_MITS", 0);
   if (p == NULL)
      pmydata->mits = MITS_DEFAULT;
   else
      pmydata->mits = atoi(p);

   p = bi_getenv("BENCHIT_NUM_CPUS", 0);
   if (p == NULL)
      pmydata->cpuCount = -1;
   else
      pmydata->cpuCount = atoi(p);
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   char *p = 0;
   char buff[80];
   mydata_t *penv = NULL;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence =
      bi_strdup("for (j = js; j < je; j++)#"
                " for (i = is; i < ie; i++) {#"
                "  index = j * MAXN + i;#"
                "  a[index] = 0.25 * (b[index - MAXN] + b[index - 1] + b[index + MAXN] + b[index + 1]) - h * h * f[index];#"
                "  ...#"
                "  b[index] = 0.25 * (a[index - MAXN] + a[index - 1] + a[index + MAXN] + a[index + 1]) - h * h * f[index];#"
                "  ...#"
                "  diff = a[index] - b[index];#"
                "  sum += diff * diff;#"
                " }");
   infostruct->kerneldescription =
      bi_strdup
      ("Jacobi algorithm measuring FLOPS (ij, ji) for change of dimension, for a given number of posix threads, one malloc for biggest dimension");
   if (penv->cpuCount > 0)
      sprintf(buff, "X-Y-Dimension (%d Threads, %d CPUs)", penv->threadCount,
              penv->cpuCount);
   else
      sprintf(buff, "X-Y-Dimension (%d Threads, unknown #CPUs)",
              penv->threadCount);
   infostruct->xaxistext = bi_strdup(buff);
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 1;
   infostruct->numfunctions = RUNS * NUM_FUNC;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   infostruct->yaxistexts[0] = bi_strdup("flop/s");
   infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[0] = 0;           // logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("flop/s (ij)");

   /* setting up y axis texts and properties */
   infostruct->yaxistexts[1] = bi_strdup("s");
   infostruct->selected_result[1] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[1] = 0;           // logarythmic axis 10^x
   infostruct->legendtexts[1] = bi_strdup("time (ij)");

   /* setting up y axis texts and properties */
   infostruct->yaxistexts[2] = bi_strdup("flop/s");
   infostruct->selected_result[2] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[2] = 0;           // logarythmic axis 10^x
   infostruct->legendtexts[2] = bi_strdup("flop/s (ji)");

   /* setting up y axis texts and properties */
   infostruct->yaxistexts[3] = bi_strdup("s");
   infostruct->selected_result[3] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[3] = 0;           // logarythmic axis 10^x
   infostruct->legendtexts[3] = bi_strdup("time (ji)");

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
void *bi_init(int problemSizemax) {
   mydata_t *pmydata = NULL;
   myinttype maxsize = 0;

   pmydata = (mydata_t *) malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);

   maxsize = (myinttype) bi_get_list_maxelement();
   pmydata->maxsize = maxsize;

   pmydata->a =
      (double *)malloc((maxsize + 2) * (maxsize + 2) * sizeof(double));
   if (pmydata->a == NULL) {
      printf("No more core\n");
      exit(127);
   }
   pmydata->b =
      (double *)malloc((maxsize + 2) * (maxsize + 2) * sizeof(double));
   if (pmydata->b == NULL) {
      printf("No more core\n");
      exit(127);
   }
   pmydata->f =
      (double *)malloc((maxsize + 2) * (maxsize + 2) * sizeof(double));
   if (pmydata->f == NULL) {
      printf("No more core\n");
      exit(127);
   }
   pmydata->diffnormArray =
      (double *)malloc(pmydata->threadCount * sizeof(double));
   if (pmydata->diffnormArray == NULL) {
      printf("No more core\n");
      exit(127);
   }

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
int bi_entry(void *mdpv, int iproblemSize, double *dresults) {
   /* dstart, dend: the start and end time of the measurement */
   /* dtime: the time for a single measurement in seconds */
   double dstart = 0.0, dend = 0.0, dtime[RUNS], flop[RUNS];
   /* ii, tc (thread_count) is used for loop iterations */
   myinttype ii = 0, tc = 0, imyproblemSize = 0;
   /* cast void* pointer */
   mydata_t *pmydata = (mydata_t *) mdpv;

   double sweep2dflop = 0.0, diff2dflop = 0.0;
   thread_t *tp = NULL;
   myinttype status = 0;

   IDL(1, printf("reached function bi_entry\n"));

   /* get current problem size from problemlist */
   pmydata->maxn = (myinttype) bi_get_list_element(iproblemSize);
   pmydata->maxn += 2;
   pmydata->nxy = (pmydata->maxn - 2) / sqrt(pmydata->threadCount);
   /* pmydata->stripes = (pmydata->maxn - 2) / pmydata->nxy;
    * this cause fp-exceptions for small maxn -> nxy = 0
    * workaround works, but I don't know if the result is correct
    */
   if (pmydata->nxy != 0)
      pmydata->stripes = (pmydata->maxn - 2) / pmydata->nxy;
   else
      pmydata->stripes = 1;
   pmydata->h = 1.0 / ((double)(pmydata->nxy + 1));

   sweep2dflop = 7 * (double)(pmydata->maxn - 2) * (double)(pmydata->maxn - 2);
   diff2dflop = 3 * (double)(pmydata->maxn - 2) * (double)(pmydata->maxn - 2);

   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL)
      return 1;

   for (ii = 0; ii < RUNS; ii++) {
      /* initialize threads */
      pmydata->thread = (void *)malloc(pmydata->threadCount * sizeof(thread_t));
      if (pmydata->thread == NULL) {
         printf("No more core\n");
         exit(127);
      }
      tp = (thread_t *) pmydata->thread;
      barrier_init(&pmydata->barrier, pmydata->threadCount);

      /* initialize matricies */
      twodinit(pmydata);

      /* get the actual time do the measurement / your algorythm get the actual 
       * time */
      switch (ii) {
         case 0:
            dstart = bi_gettime();     /* start time */
            for (tc = 0; tc < pmydata->threadCount; tc++) {
               tp[tc].number = tc;
               tp[tc].mdp = pmydata;
               status =
                  pthread_create(&tp[tc].thread_id, NULL,
                                 jacobi_thread_routine_ij, (void *)&tp[tc]);
               if (status != 0)
                  err_abort(status, "Create thread");
            }
            for (tc = 0; tc < pmydata->threadCount; tc++) {
               status = pthread_join(tp[tc].thread_id, NULL);
               if (status != 0)
                  err_abort(status, "Join thread");
            }
            dend = bi_gettime();       /* end time */
            break;
         case 1:
         default:
            dstart = bi_gettime();     /* start time */
            for (tc = 0; tc < pmydata->threadCount; tc++) {
               tp[tc].number = tc;
               tp[tc].mdp = pmydata;
               status =
                  pthread_create(&tp[tc].thread_id, NULL,
                                 jacobi_thread_routine_ji, (void *)&tp[tc]);
               if (status != 0)
                  err_abort(status, "Create thread");
            }
            for (tc = 0; tc < pmydata->threadCount; tc++) {
               status = pthread_join(tp[tc].thread_id, NULL);
               if (status != 0)
                  err_abort(status, "Join thread");
            }
            dend = bi_gettime();       /* end time */
      }

      /* calculate the used time and #Operations */
      dtime[ii] = dend - dstart;
      dtime[ii] -= dTimerOverhead;
      flop[ii] = (double)pmydata->mitsdone * (2.0 * sweep2dflop + diff2dflop);

      /* cleanup threads */
      barrier_destroy(&pmydata->barrier);
      free(pmydata->thread);
   }

   dresults[0] = (double)(pmydata->maxn);
   for (ii = 0; ii < 2; ii++) {
      /* If the operation was too fast to be measured by the timer function,
       * mark the result as invalid */
      if (dtime[ii] < dTimerGranularity)
         dtime[ii] = INVALID_MEASUREMENT;

      /* store the results in results[1], results[2], ... [1] for the first
       * function, [2] for the second function and so on ... the index 0 always
       * keeps the value for the x axis */
      dresults[ii * 2 + 1] =
         (dtime[ii] !=
          INVALID_MEASUREMENT) ? flop[ii] / dtime[ii] : INVALID_MEASUREMENT;
      dresults[ii * 2 + 2] =
         (dtime[ii] != INVALID_MEASUREMENT) ? dtime[ii] : INVALID_MEASUREMENT;
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
      if (pmydata->a)
         free(pmydata->a);
      if (pmydata->b)
         free(pmydata->b);
      if (pmydata->f)
         free(pmydata->f);
      /* free diffnorm */
      if (pmydata->diffnormArray)
         free(pmydata->diffnormArray);

      free(pmydata);
   }

   return;
}

