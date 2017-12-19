/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/BLAS3/C/0/MKL/sgemm/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Measurment of sgemm performance
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"
#include "aligned_memory.h"

#ifdef USE_MKL
  #include <mkl_cblas.h>
  #define BLASOP cblas_sgemm	/* blas operation */
#endif
#ifdef USE_ACML
  #include <acml.h>
  #define BLASOP sgemm		/* blas operation */
#endif
#ifdef USE_ATLAS
  #include <cblas.h>
  #define BLASOP cblas_sgemm	/* blas operation */
#endif

#define DT float		/* datatype */
#define ALIGN 16		/* alignment */
#define EPSILON 1.0e-6		/* epsilon for result validation, 1.0e-6 for single and 1.0e-15 for double */

#define KERNELDES "sgemm"
#define LEGENDY "FLOPS (sgemm)"

#ifndef __work_h
#define __work_h

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif


/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata
{
   myinttype maxsize;
   DT *A, *B, *C;
} mydata_t;

#endif


/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   char * p = 0;
   mydata_t * penv;
   
   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

   /* get environment variables for the kernel */
   infostruct->codesequence = bi_strdup("start kernel; matrix-matrix multiplication; ");
   infostruct->kerneldescription = bi_strdup(KERNELDES);
   infostruct->xaxistext = bi_strdup("Problem Size");
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
   infostruct->base_yaxis[0] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup(LEGENDY);
 
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
  myinttype i, m, tmp;

  pmydata = (mydata_t*)malloc(sizeof(mydata_t));
  if (pmydata == 0)
  {
     fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
     exit(127);
  }

  m = (myinttype)bi_get_list_element(1);
  for(i=2; i<=problemSizemax; i++){
    tmp = (myinttype)bi_get_list_element(i);
    if(tmp>m) m = tmp;
  }
  pmydata->maxsize = m;
  
  IDL(3, printf("\nAllocate (%d,%d)-Matrix\n",m,m));

  pmydata->A = (DT*)_aligned_malloc(m*m * sizeof(DT), ALIGN);
  pmydata->B = (DT*)_aligned_malloc(m*m * sizeof(DT), ALIGN);
  pmydata->C = (DT*)_aligned_malloc(m*m * sizeof(DT), ALIGN);
  
  /* matrix A is stored column by column like in fortran
   * elseif you must call cblas_sgemm with "CblasRowMajor" and thru that the algo
   * first transpose the matrix -> less performance
   */
  for(i=0; i<m*m; i++){
    pmydata->A[i] = 0.5;
    pmydata->B[i] = 2.0;
  }

  return (void *)pmydata;
}


/** initialize array with 0
 */
void initzero(DT *array, int size)
{
  myinttype i;

  for(i=0; i<size; i++){
    array[i] = 0.0;
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
int bi_entry(void * mdpv, int iproblemSize, double * dresults)
{
  /* dstart, dend: the start and end time of the measurement */
  /* dtime: the time for a single measurement in seconds */
  double dstart = 0.0, dend = 0.0, dtime = 0.0;
  /* flops stores the calculated FLOPS */
  double dres = 0.0, dm = 0.0;
  /* ii is used for loop iterations */
  myinttype imyproblemSize;
  /* cast void* pointer */
  mydata_t * pmydata = (mydata_t *) mdpv;
#ifdef USE_ACML
  char trans = 'N';
#endif
  
  DT alpha=1.0;
  myinttype m, incxy=1;

  /* get current problem size from problemlist */
  imyproblemSize = (myinttype)bi_get_list_element(iproblemSize);
  m = imyproblemSize;

  /* check wether the pointer to store the results in is valid or not */
  if (dresults == NULL) return 1;

  initzero(pmydata->C, pmydata->maxsize*pmydata->maxsize);

  /* get the actual time
   * do the measurement / your algorythm
   * get the actual time
   */
  dstart = bi_gettime(); 
#ifdef USE_ACML
  BLASOP(trans, trans, m, m, m, alpha, pmydata->A, m, pmydata->B, m, alpha, pmydata->C, m);
#else
  BLASOP(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, alpha, pmydata->A, m, pmydata->B, m, alpha, pmydata->C, m);
#endif
  dend = bi_gettime();
  
  if(abs((pmydata->C)[0]-(DT)m) > abs((DT)m)*EPSILON || abs((pmydata->C)[m*m-1]-(DT)m) > abs((DT)m)*EPSILON) {
    IDL(0, printf("ERROR: the result is not valid!\n"));
    IDL(0, printf("expected: C(1,1)=%e, C(m,m)=%e,   got: C(1,1)=%e, C(m,m)=%e\n",(DT)m,(DT)m,(pmydata->C)[0],(pmydata->C)[m*m-1]));
  }

  /* calculate the used time and FLOPS */
  dtime = dend - dstart;
  dtime -= dTimerOverhead;
  dm = (double)m;
  dres  = dm * dm * dm + (dm-1) * dm * dm;	/* m*m*m mult and m-1*m*m add for A*B */
  dres += dm * dm;				/* m*m mult for alpha */
  dres += 2.0 * dm * dm;			/* m*m mult for C and m*m add for final result */
  
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
  dresults[1] = (dtime!=INVALID_MEASUREMENT) ? dres / dtime : INVALID_MEASUREMENT;

  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   mydata_t * pmydata = (mydata_t*)mdpv;

   _aligned_free(pmydata->A);
   _aligned_free(pmydata->B);
   _aligned_free(pmydata->C);

   if (pmydata) free(pmydata);
   return;
}

