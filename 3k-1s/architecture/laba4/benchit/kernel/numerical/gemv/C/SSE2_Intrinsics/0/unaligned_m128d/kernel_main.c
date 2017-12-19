/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/gemv/C/SSE2_Intrinsics/0/unaligned_m128d/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: C DGEMV kernel (SSE2, unaligned data)
 *******************************************************************/

#include <xmmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"

/*  Header for local functions
 */
#include "work.h"

/** These variables will help us to keep the overview over the arrays
  * we access for our functions/data.
  */
/* Number of different ways an algorithm will be measured.
   normal, blas, sse
  */
int functionCount;
/* Number of fixed functions we have per measurement.
   execution time and MFLOPS are measured for each loop order
   -> valuesPerFunction=2 */
int valuesPerFunction;

void initData(mydata_t * mds, int n) {
   int i, j;
   for (i = 0; i < n; i++) {
      mds->a[i] = 1.1;
      mds->y[i] = 1.1;
      for (j = 0; j < n; j++)
         mds->x[i * n + j] = 0.01;
   }
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   int i = 0, j = 0;                   /* loop var for functionCount */
   char *p = 0;


   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

   infostruct->codesequence =
      bi_strdup
      ("for (i=0;i<sizeVector;i++)#  for (j=0;j<sizeAusgabe;j++)#    y[j]=alpha*a[i]*x[i*sizeAusgabe+j]+y[j];");
   infostruct->xaxistext = bi_strdup("Matrix Size");
   infostruct->kerneldescription =
      bi_strdup("Matrix Vector Multiply (C + SSE2)");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   /* B ######################################################## */
   functionCount = 2;                     /* number versions of this algorithm
                                        * (norm,blas,sse_,sse2_(algn)= 4 */
   valuesPerFunction = 1;       /* MFLOPS (calculated) */
   /* ######################################################## */
   infostruct->numfunctions = functionCount * valuesPerFunction;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   for (j = 0; j < functionCount; j++) {
      /* B ######################################################## */
      int index1 = 0 * functionCount + j;
      // 1st function
      infostruct->yaxistexts[index1] = bi_strdup("FLOPS");
      infostruct->selected_result[index1] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[index1] = 0;
      switch (j) {
            /* B ######################################################## */
         case 0:
            infostruct->legendtexts[index1] = bi_strdup("FLOPS (ij)");
            break;
         case 1:
            infostruct->legendtexts[index1] = bi_strdup("FLOPS (ji)");
            break;
         default:
            fprintf(stderr, "Should not reach default section of case.\n");
            fflush(stderr);
            exit(127);
      }
   }
   if (DEBUGLEVEL > 3) {
      /* the next for loop: */
      /* this is for your information only and can be ereased if the kernel
       * works fine */
      for (i = 0; i < infostruct->numfunctions; i++) {
         printf("yaxis[%2d]=%s\t\t\tlegend[%2d]=%s\n", i,
                infostruct->yaxistexts[i], i, infostruct->legendtexts[i]);
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
   myinttype maxsize;
   mydata_t *mdp;

   mdp = (mydata_t *) malloc(sizeof(mydata_t));
   if (mdp == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }

   maxsize = (myinttype) bi_get_list_maxelement();
   mdp->maxsize = maxsize;

   mdp->x = (float*) malloc((maxsize * maxsize) * sizeof(double));
   IDL(3, printf("Alloc 1 done\n"));
   mdp->y = (float*) malloc((maxsize) * sizeof(double));
   IDL(3, printf("Alloc 2 done\n"));
   mdp->a = (float*) malloc((maxsize) * sizeof(double));
   IDL(3, printf("Alloc 3 done\n"));

   if ((mdp->a == 0) || (mdp->x == 0) || (mdp->y == 0)) {
      printf("malloc (%ld bytes) failed in bi_init()\n",
             (long)((2.0 + maxsize) * maxsize * sizeof(double)));
      bi_cleanup(mdp);
      exit(127);
   }

   IDL(2, printf("leave bi_init\n"));

   return (void *)mdp;
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
int bi_entry(void *mdpv, int iproblemSize, double *results) {
   double start = 0.0;
   double stop = 0.0;
   // used for precision
   long numberOfRuns = 1, i = 0;
   myinttype j = 0, imyproblemSize = 0;

   /* calculate real problemSize */
   imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);

   mydata_t *mdp = (mydata_t *) mdpv;

   /* check wether the pointer to store the results in is valid or not */
   if (results == NULL)
      return 1;

   /* B ######################################################## */
   /* maybe some init stuff in here */
   initData(mdpv, imyproblemSize);
   /* ######################################################## */

   for (j = 0; j < functionCount; j++) {
      // reset variables
      numberOfRuns = 1;
      start = 0.0;
      stop = 0.0;
      /* B ######################################################## */
      int index1 = 0 * functionCount + j;
      /* choose version of algorithm */
      switch (j) {
         case 1:
            do {
               initData(mdpv, imyproblemSize);
               start = bi_gettime();
               for (i = 0; i < numberOfRuns; i++) {
                  sseunalignJI_(imyproblemSize, imyproblemSize, 1.1, 0.9,
                                mdp->a, mdp->x, mdp->y);
               }
               stop = bi_gettime();
               stop = stop - start - dTimerOverhead;
               numberOfRuns = numberOfRuns * 8;
            } while (stop < 0.01);
            numberOfRuns = (long)(numberOfRuns / 8);
            stop = stop / ((1.0) * (numberOfRuns));
            break;
         case 0:
            do {
               initData(mdpv, imyproblemSize);
               start = bi_gettime();
               for (i = 0; i < numberOfRuns; i++)
                  sseunalignIJ_(imyproblemSize, imyproblemSize, 1.1, 0.9,
                                mdp->a, mdp->x, mdp->y);
               stop = bi_gettime();
               stop = stop - start - dTimerOverhead;
               numberOfRuns = numberOfRuns * 8;
            } while (stop < 0.01);
            numberOfRuns = (long)(numberOfRuns / 8);
            stop = stop / ((1.0) * (numberOfRuns));
            break;
      }
      /* store the results in results[1], results[2], ...
      * [1] for the first function, [2] for the second function
      * and so on ...
      * the index 0 always keeps the value for the x axis
      */
      /* B ########################################################*/
      // the xaxis value needs to be stored only once!
      if (j == 0)
         results[0] = (double)imyproblemSize;
      results[index1 + 1] =
         (2.0 * imyproblemSize + 2.0 * imyproblemSize * imyproblemSize) / stop;
      /* ######################################################## */
   }

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void *mdpv) {
   mydata_t *mdp = (mydata_t *) mdpv;
   /* may be freeing our arrays here */
   if (mdp->a)
      free(mdp->a);
   if (mdp->x)
      free(mdp->x);
   if (mdp->y)
      free(mdp->y);
   if (mdp)
      free(mdp);
   return;
}

