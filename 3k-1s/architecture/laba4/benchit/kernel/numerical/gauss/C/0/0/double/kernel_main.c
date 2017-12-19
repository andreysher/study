/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/gauss/C/0/0/double/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Gaussian Linear Equation System Solver
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gauss.h"

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
      = bi_strdup("for(i=0; i<n-1; i++)#"
                  "  for(j=0; j<n; j++){#"
                  "    b[j]-=b[i]*(A[j][i]/A[i][i]);#"
                  "    for(k=0; k<n; k++)#"
                  "    {#"
                  "      A[j][k]-=(A[j][i]/A[i][i])*A[i][k];#"
                  "    }}#"
                  "x[n-1]=b[n-1]/A[n-1][n-1];#"
                  "for(i=n-2;i>=0;i--){#"
                  "  x[i]=b[i];#"
                  "  for(j=i+1;j<n;j++){#"
                  "    x[i]-=A[i][j]*x[j];#"
                  "  };#"
                  "  x[i]/=A[i][i];#"
                  "};");
   infostruct->kerneldescription
      = bi_strdup("Gaussian Linear Equation System Solver");
   infostruct->xaxistext = bi_strdup("Number of unknown variables");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 2;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   infostruct->yaxistexts[0] = bi_strdup("flop/s");
   infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[0] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("flop/s");

   /* setting up y axis texts and properties */
   infostruct->yaxistexts[1] = bi_strdup("s");
   infostruct->selected_result[1] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[1] = 0; //logarythmic axis 10^x
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
void* bi_init(int problemSizemax) {
   mydata_t * pmydata;
   myinttype i, m, tmp;
   
   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }
   
   m = (myinttype)bi_get_list_element(1);
   for(i=2; i<=problemSizemax; i++){
      tmp = (myinttype)bi_get_list_element(i);
      if(tmp>m) m = tmp;
   }
   pmydata->maxsize = m;
   
   pmydata->A = (FPT**)malloc(m * sizeof(FPT*));
   pmydata->x = (FPT*)malloc(m * sizeof(FPT));
   pmydata->b = (FPT*)malloc(m * sizeof(FPT));
   
   if(pmydata->A==0 || pmydata->b==0 || pmydata->x==0){
      printf("malloc (%ld bytes) failed in bi_init()\n", 
         (long) ((2.0*m+m*m) * sizeof(FPT)));
      bi_cleanup((void *)pmydata);
      exit(127);
   }
   
   for (i=0; i<m; i++) {
      pmydata->A[i] = (FPT*)malloc(m * sizeof(FPT));
      if(pmydata->A[i]==0){
         printf("malloc (%ld bytes) failed in bi_init()\n",
            (long)(m*sizeof(FPT)));
         bi_cleanup((void *)pmydata);
         exit(127);
      }
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
int bi_entry(void * mdpv, int iproblemSize, double * dresults) {
   /* dstart, dend: the start and end time of the measurement */
   /* dtime: the time for a single measurement in seconds */
   double dstart = 0.0, dend = 0.0, dtime = 0.0, flop = 0.0;
   /* i,j,k is used for loop iterations */
   myinttype i=0, j=0, k=0, imyproblemSize, m, err=0;
   /* cast void* pointer */
   mydata_t * pmydata = (mydata_t *) mdpv;
   
   /* tmp and dummy variables for A,x,b */
   double **A=pmydata->A, *b=pmydata->b, *x=pmydata->x; 
   double dummy_x[pmydata->maxsize];
   /* result verification */
   double sum=0.0;

   /* get current problem size from problemlist */
   imyproblemSize = (myinttype)bi_get_list_element(iproblemSize);
   m = imyproblemSize;

   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL)
      return 1;

   IDL(3, printf("\ninitializing the matrix A\n"));
   err = init_mat_c(pmydata->A, m, m, pmydata->b, m, pmydata->x, m, 0, m-1, 0, m-1);
   if (err!=0) {
      printf("error while initializing the problem matrix\n");
      exit(127);
   }

   /* get the actual time
    * do the measurement / your algorythm
    * get the actual time
    */
   IDL(3,printf("Starting run")); 
   dstart = bi_gettime();
/******the calculation part*****************************/
   for (i=0; i<m-1; i++) {
/* perform triangulation in this step */
      for (j=i+1; j<m; j++){
         b[j] -= b[i]*(A[j][i]/A[i][i]);
         for (k=i+1; k<m; k++){
            A[j][k] -= (A[j][i]/A[i][i])*A[i][k];
         }
      }
   }
/* resubstitut the x vektor */
   x[m-1] = b[m-1]/A[m-1][m-1];
   for (i=m-2; i>=0; i--) {
      x[i] = b[i];
      for (j=i+1; j<m; j++) {
         x[i] -= A[i][j]*x[j];
      }
      x[i] /= A[i][i];
   }
/*******************************************************/
   dend = bi_gettime();
   IDL(3, printf(" (OK)\n"));

   /* calculate the used time and #Operations */
   dtime = dend - dstart;
   dtime -= dTimerOverhead;

   flop = 3.0*(((double)(m-1)*(double)m)/2.0);           // b[j] 2. inner loop
   flop+= 3.0*(((double)(m-1)*(double)m*(double)m)/4.0); // A[j][k] 3. inner loop
   flop+= 1;                                             // x[m-1]
   flop+= 2.0*(((double)(m-1)*(double)m)/2.0);           // x[i] 2. inner loop
   flop+= (double)(m-1);                                 // x[i] 1. inner loop

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
   dresults[1] = (dtime!=INVALID_MEASUREMENT) ? flop / dtime
      : INVALID_MEASUREMENT;
   dresults[2] = (dtime!=INVALID_MEASUREMENT) ? dtime : INVALID_MEASUREMENT;

/* verification of correctness */
   IDL(3, printf("Verifying correctness of result\n"));
   err = init_mat_c(A, m, m, b, m, dummy_x, m, 0, m-1, 0, m-1);
   if (err!=0) {
      printf("error while initializing the problem matrix\n");
      exit(127);
   }
   k = 0;
   for (i=0; i<m; i++) {
      sum = 0;
      for (j=0; j<m; j++) {
         IDL(3, printf("%g ",(A[i])[j]));
         sum += A[i][j]*x[j];
      }
      k += (abs(sum-b[i])>TOLERANCE);
      IDL(3, printf("   %g      %g     %g   %d\n", x[i], b[i], sum, k));
   }
   if (k!=0) printf("BenchIT: gauss_c: ERROR: The result was not within the tolerance. Aborting...");

   return (k);
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv) {
   mydata_t * pmydata = (mydata_t*)mdpv;
   myinttype i;

   if (pmydata) {
      /* free matrix A */
      if (pmydata->A) {
         for (i=0; i<pmydata->maxsize; i++) {
            if (pmydata->A[i])
               free(pmydata->A[i]);
         }
         free(pmydata->A);
      }
      /* free vector b, x */
      if (pmydata->b)
         free(pmydata->b);
      if (pmydata->x)
         free(pmydata->x);

      free(pmydata);
   }

   return;
}

