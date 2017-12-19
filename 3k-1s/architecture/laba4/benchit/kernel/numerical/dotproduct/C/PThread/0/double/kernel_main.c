/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/dotproduct/C/PThread/0/double/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for dot product of two vectors with posix threads
 *******************************************************************/

#include "dotproduct.h"

/* Reads the environment variables used by this kernel. 
 * see interface.h for bi_getenv("name", exit_on_error)
 */
void evaluate_environment(mydata_t * pmydata) {
   char *p = 0;

   /* add additional parameters, except BENCHIT_KERNEL_PROBLEMLIST from the
    * parameters file BENCHIT_KERNEL_PROBLEMLIST will be done in bi_getinfo */
   p = bi_getenv("BENCHIT_KERNEL_PTHREAD_THREADS_COUNT_START", 0);
   if (p == NULL)
      pmydata->threadsCountStart = THREADS_COUNT_START_DEFAULT;
   else
      pmydata->threadsCountStart = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_PTHREAD_THREADS_COUNT_DOUBLE", 0);
   if (p == NULL)
      pmydata->threadsCountDouble = THREADS_COUNT_DOUBLE_DEFAULT;
   else
      pmydata->threadsCountDouble = atoi(p);

   pmydata->threadsCountMax = pmydata->threadsCountStart;
   pmydata->threadsCountMax <<= pmydata->threadsCountDouble;
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
   myinttype a = 0, b = 0;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence =
      bi_strdup("create threads#"
                "execute threads:#"
                "  do i=1,n#"
                "    sum=sum+x(i)*y(i)#"
                "  enddo#"
                "kill threads#"
                "print exexution time per thread");
   infostruct->kerneldescription =
      bi_strdup("Core for dot product of two vectors with posix threads");
   infostruct->xaxistext = bi_strdup("vector length");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 1;
   infostruct->numfunctions = penv->threadsCountDouble + 1;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
      infostruct->yaxistexts[a] = bi_strdup("s");
      infostruct->selected_result[a] = SELECT_RESULT_LOWEST;
      infostruct->base_yaxis[a] = 0;        // logarythmic axis 10^x
      sprintf(buff, "%d threads", b);
      infostruct->legendtexts[a] = bi_strdup(buff);
      b <<= 1;
   }

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
   myinttype ii = 0, m = 0;
   double **mem = NULL;

   pmydata = (mydata_t *) malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);

   m = (myinttype) bi_get_list_maxelement();
   pmydata->maxsize = m;

   IDL(1,
       printf("\nneed %d MByte memory ...\n",
              (int)((2 * m * sizeof(double)) / (1024 * 1024))));

   mem = (double **)malloc(2 * sizeof(double *));
   mem[0] = (double *)malloc(pmydata->maxsize * sizeof(double));
   mem[1] = (double *)malloc(pmydata->maxsize * sizeof(double));
   if ((mem == NULL) || (mem[0] == NULL) || (mem[1] == NULL)) {
      printf("Can't alloc mem, not enough memory\n");
      exit(127);
   }
   for (ii = 0; ii < pmydata->maxsize; ii++) {
      mem[0][ii] = 0.5;
      mem[1][ii] = 2.0;
   }
   pmydata->mem = mem;

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
   double *dstart = NULL, *dend = NULL, *dtime = NULL;
   /* ii is used for loop iterations */
   myinttype ii = 0, imyproblemSize = 0;
   /* cast void* pointer */
   mydata_t *pmydata = (mydata_t *) mdpv;
   /* needed for kernel threads */
   double **mem = (double **)(pmydata->mem);
   int err = 0, perthread = 0, outerLoop = 0, thread_count = 0;
   pthread_t *threadids = NULL;
   double **threadmem = NULL;
   double most_threads_p_size = 0.0, last_thread_p_size = 0.0;

   IDL(2, printf("enter bi_entry\n"));

   /* alloc mem to store the used time */
   dstart =
      (double *)malloc((pmydata->threadsCountDouble + 1) * sizeof(double));
   dend = (double *)malloc((pmydata->threadsCountDouble + 1) * sizeof(double));
   dtime = (double *)malloc((pmydata->threadsCountDouble + 1) * sizeof(double));

   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL)
      return 1;

   /* get current problem size from problemlist */
   imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);

   threadmem =
      (double **)malloc(pmydata->threadsCountMax * 3 * sizeof(double *));
   if (threadmem == NULL) {
      printf("no more core while allocating memory for thread arguments\n");
      fflush(stdout);
      bi_cleanup(mdpv);
      exit(127);
   }

   IDL(2, printf("allocate memory for thread id's\n"));
   threadids =
      (pthread_t *) malloc(pmydata->threadsCountMax * sizeof(pthread_t));
   if (threadids == NULL) {
      printf("no more core while allocating thread id's\n");
      fflush(stdout);
      bi_cleanup(mdpv);
      exit(127);
   }

   /* make pointers to the start of the sub array for each thread; init the
    * number of calculations per thread */
   thread_count = pmydata->threadsCountStart;
   for (outerLoop = 0; outerLoop <= pmydata->threadsCountDouble; outerLoop++) {
      IDL(3, printf("make pointers etc. - "));
      perthread = (int)floor((double)(imyproblemSize) / (double)thread_count);
      most_threads_p_size = (double)perthread;
      last_thread_p_size =
         imyproblemSize - (most_threads_p_size * (thread_count - 1));
      IDL(2,
          printf("part: %d, last: %d\n", perthread, (int)last_thread_p_size));
      IDL(3, printf("for loop bi_entry - "));
      for (ii = 0; ii < thread_count; ii++) {
         threadmem[ii * 3] = &(mem[0][ii * perthread]);
         threadmem[ii * 3 + 1] = &(mem[1][ii * perthread]);
         if (ii != thread_count - 1)
            threadmem[ii * 3 + 2] = &most_threads_p_size;
         else
            threadmem[ii * 3 + 2] = &last_thread_p_size;
      }
      IDL(3, printf("pointers done\n"));

      dstart[outerLoop] = bi_gettime();
      /* create the threads */
      for (ii = 0; ii < thread_count; ii++) {
         err =
            pthread_create(&threadids[ii], NULL, thread_func,
                           &threadmem[ii * 3]);
         if (err != 0) {
            printf("unable to create thread number %d, error %d (%s)\n", ii,
                   err, error_text(err));
            fflush(stdout);
            free(threadmem);
            free(threadids);
            bi_cleanup(pmydata);
            exit(127);
         }
      }

      IDL(2, printf("threads created, wait for termination\n"));
      /* wait for thread termination */
      for (ii = 0; ii < thread_count; ii++) {
         err = pthread_join(threadids[ii], NULL);
         if (err != 0) {
            printf("problem joining thread number %d, error %d (%s)\n", ii, err,
                   error_text(err));
            fflush(stdout);
            free(threadmem);
            free(threadids);
            bi_cleanup(pmydata);
            exit(127);
         }
      }
      IDL(2, printf("threads terminated\n"));
      dend[outerLoop] = bi_gettime();

      thread_count <<= 1;
   }

   /* store the results in results[1], results[2], ... [1] for the first
    * function, [2] for the second function and so on ... the index 0 always
    * keeps the value for the x axis */
   dresults[0] = (double)imyproblemSize;

   for (ii = 0; ii <= pmydata->threadsCountDouble; ii++) {
      /* calculate the used time */
      dtime[ii] = dend[ii] - dstart[ii];
      dtime[ii] -= dTimerOverhead;

      /* If the operation was too fast to be measured by the timer function,
       * mark the result as invalid */
      if (dtime[ii] < dTimerGranularity)
         dtime[ii] = INVALID_MEASUREMENT;

      /* store the results in results[1], results[2], ... [1] for the first
       * function, [2] for the second function and so on ... the index 0 always 
       * keeps the value for the x axis */
      dresults[ii + 1] = (dtime[ii] != INVALID_MEASUREMENT) ? dtime[ii]
         : INVALID_MEASUREMENT;
   }

   free(threadmem);
   free(threadids);

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void *mdpv) {
   mydata_t *pmydata = (mydata_t *) mdpv;

   if (pmydata) {
      /* free mem for x, y */
      if (pmydata->mem) {
         if (pmydata->mem[0])
            free(pmydata->mem[0]);
         if (pmydata->mem[1])
            free(pmydata->mem[1]);
         free(pmydata->mem);
      }

      free(pmydata);
   }

   return;
}

