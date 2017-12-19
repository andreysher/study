/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numberTheory/fibonacci/C/0/0/long/work.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for calc of Fibonacci number (iterative / recursive)
 *******************************************************************/

#ifndef __work_h
#define __work_h

#include <stdio.h>
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

extern long recfib(long lnumber);
extern long linfib(long lnumber);

#endif

