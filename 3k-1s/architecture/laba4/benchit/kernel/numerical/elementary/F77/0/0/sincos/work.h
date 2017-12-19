/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/elementary/F77/0/0/sincos/work.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Execution time of mathematical operations SINUS / COSINUS
 *         addict to input value
 *******************************************************************/

#ifndef __work_h
#define __work_h

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

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata
{
   /* additional parameters */
   int dummy;
} mydata_t;

#ifndef FPT
#define FPT double
#endif

#define ONE(X,Y) mathop(X,Y);
#define TEN(X,Y)      ONE(X,Y) ONE(X,Y) ONE(X,Y) ONE(X,Y) ONE(X,Y) ONE(X,Y) ONE(X,Y) ONE(X,Y) ONE(X,Y) ONE(X,Y)
#define HUNDRED(X,Y)  TEN(X,Y) TEN(X,Y) TEN(X,Y) TEN(X,Y) TEN(X,Y) TEN(X,Y) TEN(X,Y) TEN(X,Y) TEN(X,Y) TEN(X,Y)
#define THOUSAND(X,Y) HUNDRED(X,Y) HUNDRED(X,Y) HUNDRED(X,Y) HUNDRED(X,Y) HUNDRED(X,Y) HUNDRED(X,Y) HUNDRED(X,Y) HUNDRED(X,Y) HUNDRED(X,Y) HUNDRED(X,Y)

#define LOOPS 1000

/*     defined (_SX)      || \ */
#if (defined (_CRAY)    || \
     defined (_SR8000)  || \
     defined (_USE_OLD_STYLE_CRAY_TYPE))
#define mathopsin_ MATHOPSIN
#define mathopcos_ MATHOPCOS
#endif

void (*mathop)(FPT *, FPT *);
extern void mathopsin_(FPT *arg1, FPT *arg2);
extern void mathopcos_(FPT *arg1, FPT *arg2);
extern void entry_(myinttype *size);

#endif

