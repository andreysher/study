/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: dotproduct.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/dotproduct/F77/0/0/double/dotproduct.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for dot product of two vectors
 *******************************************************************/

#ifndef __dotproduct_h
#define __dotproduct_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "interface.h"

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

#if (defined (_CRAY)    || \
     defined (_SR8000)  || \
     defined (_USE_OLD_STYLE_CRAY_TYPE))
#define dotproduct_ DOTPRODUCT
#define bigtime_ BIGTIME
#endif

/* default ncache size is 1MB */
#define NCACHE_DEFAULT 131072

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata {
   double *mem;
   myinttype docacheflush;
   myinttype ncache;
   /* additional parameters */
   myinttype maxsize;
} mydata_t;

extern void dotproduct_(double *x, double *y, myinttype * n, myinttype * s,
                        myinttype * numthreads, myinttype * d, myinttype * c,
                        double *minl, double *maxl, double *flopmin,
                        double *flopmax, double *acache, myinttype * ncache);

#endif

