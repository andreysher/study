/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/SSE2_Intrinsics/0/unaligned_m128d/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: SSE2 Matrix Multiply (C), unaligned data
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


void initData(mydata_t* mds,int n)
{
   int i,j;
   for (i=0;i<n;i++)
   {
      for (j=0;j<n;j++)
      {
      mds->a[i*n+j]=1.1*i;
      mds->b[i*n+j]=0.3*i;
         mds->c[i*n+j]=0.0;
      }
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
   int i = 0, j = 0; /* loop var for functionCount */


   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

  infostruct->codesequence = bi_strdup(
  					"for (i=0;i<s;i++)#"
					"{#"
					"  for (j=0;j<s-s%2;j=j+2)#"
					"  {#"
					"    xmm_c=_mm_loadu_pd(&c[i*s+j]);#"
					"    for (k=0;k<s;k++)#"
					"    {#"
					"      xmm_a=_mm_load1_pd(&a[i*s+k]);#"
					"      xmm_b=_mm_loadu_pd(&b[k*s+j]);#"
					"      xmm_temp=_mm_mul_pd(xmm_a,xmm_b);#"
					"      xmm_c=_mm_add_pd(xmm_c,xmm_temp);#"
					"    }#"
					"    mm_storeu_pd(&c[i*s+j],xmm_c);#"
					"  }#"
					"  for (j=s-s%2;j<s;j++)#"
					"    for (k=0;k<s;k++)#"
					"      c[i*s+j]=c[i*s+j]+a[i*s+k]*b[k*s+j];"
					"}#"
					);
   infostruct->xaxistext = bi_strdup("Matrix Size");
   infostruct->kerneldescription = bi_strdup("Matrix Multiply, SSE2, unaligned (C)");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   /* B ########################################################*/
   functionCount = 6; /* number versions of this algorithm (norm,blas,sse_,sse2_(algn)= 4 */
   valuesPerFunction = 1; /* MFLOPS (calculated) */
   /*########################################################*/
   infostruct->numfunctions = functionCount * valuesPerFunction;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   for (j = 0; j < functionCount; j++)
   {
      /* B ########################################################*/
      int index1 = 0 * functionCount + j;
      // 1st function
      infostruct->yaxistexts[index1] = bi_strdup("FLOPS");
      infostruct->selected_result[index1] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[index1] = 0;
      switch (j)
      {
         /* B ########################################################*/
     case 0:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("FLOPS (ijk)");
        break;
      case 1:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("FLOPS (ikj)");
        break;
      case 2:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("FLOPS (jik)");
        break;
      case 3:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("FLOPS (jki)");
        break;
      case 4:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("FLOPS (kij)");
        break;
      case 5:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("FLOPS (kji)");
        break;
      default:
        fprintf(stderr, "Should not reach default section of case.\n");
        fflush(stderr);
        exit(127);
         /*case 0:
         default:
            infostruct->legendtexts[index1] =
               bi_strdup("Normal - Calculation Time in s");
            infostruct->legendtexts[index2] =
               bi_strdup("Normal - FLOPS");
            infostruct->legendtexts[index3] =
               bi_strdup("Normal - SSE Instructions");*/
         /*########################################################*/
      }
   }
   if (DEBUGLEVEL > 3)
   {
      /* the next for loop: */
      /* this is for your information only and can be ereased if the kernel works fine */
      for (i = 0; i < infostruct->numfunctions; i++)
      {
         printf("yaxis[%2d]=%s\t\t\tlegend[%2d]=%s\n",
            i, infostruct->yaxistexts[i], i, infostruct->legendtexts[i]);
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
void* bi_init(int problemSizemax)
{
   mydata_t* mdp;
   mdp = (mydata_t*)malloc(sizeof(mydata_t));
   if (mdp == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }

  /* calculate real maximum problem size
     problemSizemax might be smaller then BENCHIT_KERNEL_PROBLEMSIZE_MAX
     if BENCHIT_KERNEL_PROBLEMSIZE_INCREMENT is greater then one */

  problemSizemax = (int)bi_get_list_maxelement();

  mdp->a=(double*)malloc((problemSizemax*problemSizemax) * sizeof(double));
  IDL(3, printf("Alloc 1 done\n"));
  mdp->b=(double*)malloc((problemSizemax*problemSizemax) * sizeof(double));
  IDL(3, printf("Alloc 2 done\n"));
  mdp->c=(double*)malloc((problemSizemax*problemSizemax) * sizeof(double));
  IDL(3, printf("Alloc 3 done\n"));
  
  if((mdp->a==0) || (mdp->b==0) || (mdp->c==0))
    {
      printf("malloc (%ld bytes) failed in bi_init()\n",
	     (long) ((3.0*problemSizemax)*problemSizemax * sizeof(double)));
      bi_cleanup(mdp);
      exit(127);
    }

  IDL(2, printf("leave bi_init\n"));

  return (void*)mdp;
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
int bi_entry(void* mdpv, int problemSize, double* results)
{
  double start=0.0;
  double stop=0.0;
  // used for precision
  long numberOfRuns=1,i=0;
  int j=0;
  
  /* calculate real problemSize */
  problemSize = (int)bi_get_list_element(problemSize);
  
   mydata_t* mdp = (mydata_t*)mdpv;

   /* check wether the pointer to store the results in is valid or not */
   if (results == NULL) return 1;

   /* B ########################################################*/
   /* maybe some init stuff in here */
   initData(mdpv,problemSize);
   /*########################################################*/

   for (j = 0; j < functionCount; j++)
   {
    // reset variables
    numberOfRuns=1;
    start=0.0;
    stop=0.0;
      /* B ########################################################*/
      int index1 = 0 * functionCount + j;
      /* choose version of algorithm */
      switch (j) {
         case 5:
                 do
                 {
                   initData(mdpv,problemSize);
                   start=bi_gettime();
                   for (i=0;i<numberOfRuns;i++)
                   {
                     multassekji_(mdp->a,mdp->b,mdp->c,&problemSize);
                   }
                   stop=bi_gettime();
                   stop=stop-start-dTimerOverhead;
                   numberOfRuns=numberOfRuns*8;
                 } while (stop<0.001);
                 numberOfRuns=(long)(numberOfRuns/8);
                 stop=stop/((1.0)*(numberOfRuns));
                 break;
         case 4:
                 do
                 {
                   initData(mdpv,problemSize);
                   start=bi_gettime();
                   for (i=0;i<numberOfRuns;i++)
                     multassekij_(mdp->a,mdp->b,mdp->c,&problemSize);
                   stop=bi_gettime();
                   stop=stop-start-dTimerOverhead;
                   numberOfRuns=numberOfRuns*8;
                 } while (stop<0.001);
                 numberOfRuns=(long)(numberOfRuns/8);
                 stop=stop/((1.0)*(numberOfRuns));
                 break;
         case 3:
                 do
                 {
                   initData(mdpv,problemSize);
                   start=bi_gettime();
                   for (i=0;i<numberOfRuns;i++)
                     multassejki_(mdp->a,mdp->b,mdp->c,&problemSize);
                   stop=bi_gettime();
                   stop=stop-start-dTimerOverhead;
                   numberOfRuns=numberOfRuns*8;
                 } while (stop<0.001);
                 numberOfRuns=(long)(numberOfRuns/8);
                 stop=stop/((1.0)*(numberOfRuns));
                 break;
         case 2:
                 do
                 {
                   initData(mdpv,problemSize);
                   start=bi_gettime();
                   for (i=0;i<numberOfRuns;i++)
                     multassejik_(mdp->a,mdp->b,mdp->c,&problemSize);
                   stop=bi_gettime();
                   stop=stop-start-dTimerOverhead;
                   numberOfRuns=numberOfRuns*8;
                 } while (stop<0.001);
                 numberOfRuns=(long)(numberOfRuns/8);
                 stop=stop/((1.0)*(numberOfRuns));
                 break;
         case 1:
                 do
                 {
                   initData(mdpv,problemSize);
                   start=bi_gettime();
                   for (i=0;i<numberOfRuns;i++)
                     multasseikj_(mdp->a,mdp->b,mdp->c,&problemSize);
                   stop=bi_gettime();
                   stop=stop-start-dTimerOverhead;
                   numberOfRuns=numberOfRuns*8;
                 } while (stop<0.001);
                 numberOfRuns=(long)(numberOfRuns/8);
                 stop=stop/((1.0)*(numberOfRuns));
                 break;
         default:;
         case 0:
                 do
                 {
                   initData(mdpv,problemSize);
                   start=bi_gettime();
                   for (i=0;i<numberOfRuns;i++)
                     multasseijk_(mdp->a,mdp->b,mdp->c,&problemSize);
                   stop=bi_gettime();
                   stop=stop-start-dTimerOverhead;
                   numberOfRuns=numberOfRuns*8;
                 } while (stop<0.001);
                 numberOfRuns=(long)(numberOfRuns/8);
                 stop=stop/((1.0)*(numberOfRuns));
      }
      /* store the results in results[1], results[2], ...
      * [1] for the first function, [2] for the second function
      * and so on ...
      * the index 0 always keeps the value for the x axis
      */
      /* B ########################################################*/
      // the xaxis value needs to be stored only once!
      if (j == 0) results[0] = (double)problemSize;
      results[index1 + 1]= (2.0*problemSize*problemSize*problemSize)/stop;
      /*########################################################*/
   }

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   mydata_t* mdp = (mydata_t*)mdpv;
   /* B ########################################################*/
   /* may be freeing our arrays here */
   if (mdp->c) free(mdp->c);
   if (mdp->b) free(mdp->b);
   if (mdp->a) free(mdp->a);
   /*########################################################*/
   if (mdp) free(mdp);
   return;
}


