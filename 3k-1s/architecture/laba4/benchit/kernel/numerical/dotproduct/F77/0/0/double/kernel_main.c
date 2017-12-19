/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/dotproduct/F77/0/0/double/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for dot product of two vectors
 *******************************************************************/

#include "dotproduct.h"

/* Reads the environment variables used by this kernel. 
 * see interface.h for bi_getenv("name", exit_on_error)
 */
void evaluate_environment(mydata_t * pmydata) {
   char *p = 0;

   /* add additional parameters, except BENCHIT_KERNEL_PROBLEMLIST from the
    * parameters file BENCHIT_KERNEL_PROBLEMLIST will be done in bi_getinfo */
   p = bi_getenv("BENCHIT_KERNEL_PERFORM_CACHEFLUSH", 0);
   if (p == NULL)
      pmydata->docacheflush = 0;
   else
      pmydata->docacheflush = atoi(p);
   if (pmydata->docacheflush != 0)
      pmydata->docacheflush = 1;
   p = bi_getenv("BENCHIT_KERNEL_NCACHE", 0);
   if (p == NULL)
      pmydata->ncache = NCACHE_DEFAULT;
   else
      pmydata->ncache = atoi(p);
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   char *p = 0;
   mydata_t *penv;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence = bi_strdup("do i=1,n#"
                                   "  sum=sum+x(i)*y(i)#"
                                   "enddo");
   infostruct->kerneldescription =
      bi_strdup("Calculate the dot product of two vectors");
   infostruct->xaxistext = bi_strdup("vector length");
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
   infostruct->yaxistexts[0] = bi_strdup("flop/s");
   infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[0] = 0;           // logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("dot product");

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
   double *mem = NULL;

   pmydata = (mydata_t *) malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);

   m = (myinttype) bi_get_list_maxelement();
   pmydata->maxsize = m;

   mem = (double*) malloc((2 * pmydata->maxsize + pmydata->ncache) * sizeof(double));
   if (mem == 0) {
      printf("Can't alloc mem, not enough memory\n");
      exit(127);
   }
   for (ii = 0; ii < (2 * pmydata->maxsize + pmydata->ncache); ii++) {
      mem[ii] = 1.0;
   }
   pmydata->mem = mem;

   return (void *)pmydata;
}

/** stuff from old kernel used for time measurement in fortran
 */
void bigtime_(double *b);
void bigtime_(double *b) {
   *b = bi_gettime();
   return;
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
   double dstart = 0.0, dend = 0.0, dtime = 0.0;
   /* needed for kernel */
   myinttype n = 0, s = 0;
   myinttype numthreads = 0, dynamic = 0, cashflush = 0, ncache = 0;
   double *x = NULL, *y = NULL, *c = NULL;
   double minl = 0.0, maxl = 0.0, flopmin = 0.0, flopmax = 0.0;

   /* cast void* pointer */
   mydata_t *pmydata = (mydata_t *) mdpv;

   /* don't ask me why, but Guido has this in original source code (BUT: no
    * effect on kernel) */
   char *envir;
   envir = getenv("OMP_DYNAMIC");
   dynamic = envir != 0 ? strcmp(envir, "TRUE") : 1;
   dynamic = dynamic != 0 ? 0 : 1;
   envir = getenv("OMP_NUM_THREADS");
   numthreads = envir != 0 ? atoi(envir) : 1;
   /* -------------------------- */

   /* get current problem size from problemlist */
   s = (myinttype) bi_get_list_element(iproblemSize);

   /* initialize structures */
   n = pmydata->maxsize;
   x = (double *)(pmydata->mem) + 0 * n;
   y = (double *)(pmydata->mem) + 1 * n;
   c = (double *)(pmydata->mem) + 2 * n;
   ncache = pmydata->ncache;

   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL)
      return 1;

   /* get the actual time do the measurement / your algorythm get the actual
    * time */
   dotproduct_(x, y, &n, &s, &numthreads, &dynamic, &cashflush, &minl, &maxl,
               &flopmin, &flopmax, c, &ncache);

   /* store the results in results[1], results[2], ... [1] for the first
    * function, [2] for the second function and so on ... the index 0 always
    * keeps the value for the x axis */
   dresults[0] = (double)s;
   dresults[1] = flopmax;

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void *mdpv) {
   mydata_t *pmydata = (mydata_t *) mdpv;

   if (pmydata) {
      /* free mem for x, y, cacheflush */
      if (pmydata->mem)
         free(pmydata->mem);

      free(pmydata);
   }

   return;
}
