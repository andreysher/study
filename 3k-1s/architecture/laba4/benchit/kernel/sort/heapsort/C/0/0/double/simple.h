/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: simple.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/sort/heapsort/C/0/0/double/simple.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#ifndef __work_h
#define __work_h

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#include "interface.h"


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
   double *doublearray;
   /* additional parameters */
   myinttype maxsize;
} mydata_t;

extern double simple( myinttype * );

void heapsortd( double *sort, long number );
int  verifyd( double *probe, long elements );


#endif

