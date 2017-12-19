/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: gauss.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/gauss/F95/0/0/double/gauss.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Gaussian Linear Equation System Solver
 *******************************************************************/

#ifndef __gauss_h
#define __gauss_h

#include "interface.h"
#include "math.h"

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

#ifndef FPT
#define FPT double
#endif

#define MINTIME 1.0e-22
#define TOLERANCE 1.0e-5

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata
{
   /* additional parameters */
   myinttype maxsize;
   FPT *A;
   FPT *b, *x;
} mydata_t;

extern int init_mat_c(FPT *A, int sa1, int sa2, FPT *b, int sb, FPT *x, int sx, int lbd1, int ubd1, int lbd2, int ubd2);
extern int entry_(FPT *A, FPT *b, FPT *x, int* problemsize);

#endif

