/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/gemv/C/OpenMP/0/double/work.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: C DGEMV kernel
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
   double *x;
   double *y;
   double *a;
   /* additional parameters */
   myinttype maxsize;
} mydata_t;

void ij_(int sizeVector, int sizeAusgabe, double alpha, double beta, double *a,
         double *x, double *y);
void ji_(int sizeVector, int sizeAusgabe, double alpha, double beta, double *a,
         double *x, double *y);

