/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/meanSquareError/C/0/0/double/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Solve a linear mean square error problem
 *******************************************************************/

#include "mse.h"

#define NUM_FUNC 2
#define NUMPOINTS_DEFAULT 20

/* Reads the environment variables used by this kernel. 
 * see interface.h for bi_getenv("name", exit_on_error)
 */
void evaluate_environment(mydata_t * pmydata) {
   char *p = 0;

   /* add additional parameters, except BENCHIT_KERNEL_PROBLEMLIST from the
    * parameters file BENCHIT_KERNEL_PROBLEMLIST will be done in bi_getinfo */
   p = bi_getenv("BENCHIT_KERNEL_NUMPOINTS", 0);
   if (p == NULL)
      pmydata->numPoints = NUMPOINTS_DEFAULT;
   else
      pmydata->numPoints = atoi(p);
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

   infostruct->codesequence =
      bi_strdup
      ("Fit a polynom a[n-1]*x^(n-1) + ... + a[1]*x + a[0] to a number of given points");
   infostruct->kerneldescription =
      bi_strdup("Solve a linear mean square error problem");
   infostruct->xaxistext = bi_strdup("Number of coeff. (approx. polynom)");
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
   infostruct->yaxistexts[0] = bi_strdup("flop/s");
   infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[0] = 0;           // logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("flop/s");

   /* setting up y axis texts and properties */
   infostruct->yaxistexts[1] = bi_strdup("s");
   infostruct->selected_result[1] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[1] = 0;           // logarythmic axis 10^x
   infostruct->legendtexts[1] = bi_strdup("time");

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
   mydata_t *pmydata;
   myinttype ii, ij, maxunknows;
   double dx, dy;

   IDL(1, printf("reached function bi_init\n"));

   pmydata = (mydata_t *) malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);

   maxunknows = (myinttype) bi_get_list_maxelement();
   pmydata->maxUnknowns = maxunknows;

   if (pmydata->numPoints < pmydata->maxUnknowns) {
      printf("Illegal Parameters, please check PARAMETERS file\n");
      exit(127);
   }

   if (pmydata->maxUnknowns > 300) {
      printf("WARNING: The number of unknowns may be too large.\n");
      printf("An overflow of a double variable is possible!\n");
      printf
         ("Reduce maximal value in PROBLEMLIST to be sure the kernel passes through\n");
   }

   srand((unsigned int)(time(NULL)));
   
   pmydata->ppdmatrix =
      create2Darray(pmydata->numPoints, pmydata->maxUnknowns + 1);
   dx = -COEFF / 200;

/* creating matrix that represents the system of equations */
   IDL(2,
       printf("reached begin of the loop that creates system of equations\n"));
   for (ii = 0; ii < pmydata->numPoints; ii++) {
      IDL(3, printf("i=%d ", ii));
      dx = dx + COEFF / (100 * pmydata->numPoints);
      dy = 0;
      for (ij = 0; ij < pmydata->maxUnknowns; ij++) {
         pmydata->ppdmatrix[ii][ij] = pow(dx, ij);
         dy = dy + (COEFF / 200 - 2.5) * pow(dx, ij);
      }
      dy = dy + COEFF / 20000 + dy * COEFF / 50000;
      pmydata->ppdmatrix[ii][pmydata->maxUnknowns] = dy;
   }
   IDL(2, printf("reached end of the loop that creates system of equations\n"));

   IDL(1, printf("completed function bi_init\n"));

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
   double dstart = 0.0, dend = 0.0, dtime = 0.0, flop = 0.0;
   /* ii, ij is used for loop iterations */
   myinttype ii = 0, ij = 0, imyproblemSize;
   /* cast void* pointer */
   mydata_t *pmydata = (mydata_t *) mdpv;

   /* tmp and dummy variables */
   double **ppdmatrix, **ppdsoe, **ppdq, *pdsolution;
   double m;

   IDL(1, printf("reached function bi_entry\n"));

   /* get current problem size from problemlist */
   imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);
   m = imyproblemSize;

   ppdsoe = (double **)pmydata->ppdmatrix;
   ppdmatrix = create2Darray(pmydata->numPoints, imyproblemSize + 1);

   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL)
      return 1;

/* system of equations is copied to another array because the
 * following functions will change numbers but old numbers are
 * still needed for next run */
   IDL(2,
       printf("reached begin of the loop that copies system of equations\n"));
   for (ii = 0; ii < pmydata->numPoints; ii++) {
      for (ij = 0; ij < imyproblemSize; ij++) {
         ppdmatrix[ii][ij] = ppdsoe[ii][ij];
      }
   }
   IDL(2, printf("reached end of the loop that copies system of equations\n"));

/* copying last column = y-coordinates of the equations */
   for (ii = 0; ii < pmydata->numPoints; ii++) {
      ppdmatrix[ii][imyproblemSize] = ppdsoe[ii][pmydata->maxUnknowns];
   }

/* beginning time measurement */
   IDL(2, printf("reached begin of the loop that solves the equation\n"));
   dstart = bi_gettime();
   for (ii = 0; ii < imyproblemSize; ii++) {
      ppdq = createQ(ppdmatrix, pmydata->numPoints, ii);
      ppdmatrix =
         matmul(ppdq, pmydata->numPoints - ii, pmydata->numPoints - ii,
                ppdmatrix, pmydata->numPoints, imyproblemSize + 1, ii);
      /* ppdq and ppdmatrix are freed in matmul! */
   }
   /* ppdmatrix is a upper triangular matrix now */
   pdsolution = solve(ppdmatrix, imyproblemSize);
   dend = bi_gettime();
   IDL(2, printf("reached begin of the loop that solves the equation\n"));
/* finishing time measurement */

   /* calculate the used time and #Operations */
   dtime = dend - dstart;
   dtime -= dTimerOverhead;
   m = (double)imyproblemSize;
   flop = pow((double)pmydata->numPoints, 2) * m * (m + 7);
   flop -= ((double)pmydata->numPoints * m * (2 * pow(m, 2) + 18 * m - 29)) / 3;
   flop += (m * (pow(m, 3) + 12 * pow(m, 2) - 22 * m + 39)) / 6;

   free2Darray(ppdmatrix, pmydata->numPoints);
   free(pdsolution);

   /* If the operation was too fast to be measured by the timer function, mark
    * the result as invalid */
   if (dtime < dTimerGranularity)
      dtime = INVALID_MEASUREMENT;

   /* store the results in results[1], results[2], ... [1] for the first
    * function, [2] for the second function and so on ... the index 0 always
    * keeps the value for the x axis */
   dresults[0] = (double)imyproblemSize;
   dresults[1] =
      (dtime != INVALID_MEASUREMENT) ? flop / dtime : INVALID_MEASUREMENT;
   dresults[2] = (dtime != INVALID_MEASUREMENT) ? dtime : INVALID_MEASUREMENT;

   IDL(1, printf("completed function bi_entry\n"));

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void *mdpv) {
   mydata_t *pmydata = (mydata_t *) mdpv;

   if (pmydata) {
      /* free ppdmatrix */
      if (pmydata->ppdmatrix)
         free2Darray(pmydata->ppdmatrix, pmydata->numPoints);

      free(pmydata);
   }
   return;
}

