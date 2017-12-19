/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: dotproduct.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/dotproduct/C/PThread/0/double/dotproduct.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for dot product of two vectors with posix threads
 *******************************************************************/

#ifndef __dotproduct_h
#define __dotproduct_h

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <pthread.h>
#include <math.h>

#include "interface.h"
#include "errors.h"

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
   double **mem;
   myinttype threadsCountStart;
   myinttype threadsCountDouble;
   myinttype threadsCountMax;
   /* additional parameters */
   myinttype maxsize;
} mydata_t;

#define THREADS_COUNT_START_DEFAULT 2
#define THREADS_COUNT_DOUBLE_DEFAULT 4

void *thread_func(void *mcb);

#endif

