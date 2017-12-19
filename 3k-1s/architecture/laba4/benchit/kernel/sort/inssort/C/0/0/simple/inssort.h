/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: inssort.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/sort/inssort/C/0/0/simple/inssort.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Evaluate time to sort int/float array with insertion sort
 *******************************************************************/

#ifndef __inssort_h
#define __inssort_h

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
   myinttype maxsize;
   int *intarray;
   float *floatarray;
} mydata_t;

extern void inssorti( int *sort, long number );
extern void inssortf( float *sort, long number );
extern int  verifyi( int *probe, long elements );
extern int  verifyf( float *probe, long elements );

#endif

