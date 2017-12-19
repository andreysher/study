/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/gemv/C/SSE_Intrinsics/0/aligned_m128/work.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: C SGEMV kernel (SSE, aligned data)
 *******************************************************************/

#ifndef __work_h
#define __work_h
#endif

/**  to make it easier to print some messages to stdout depending
 *   on a selectable debug level
 */
#if(!defined(DEBUGLEVEL))
#define DEBUGLEVEL (0)
#endif

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
typedef struct mydata {
   float *x;
   float *y;
   float *a;
   /* additional parameters */
   myinttype maxsize;
} mydata_t;

void ssealignIJ_(int sizeVector,int sizeAusgabe,float alpha,float beta,
                 float *a, float *x, float *y);
void ssealignJI_(int sizeVector,int sizeAusgabe,float alpha,float beta,
                 float *a, float *x, float *y);

