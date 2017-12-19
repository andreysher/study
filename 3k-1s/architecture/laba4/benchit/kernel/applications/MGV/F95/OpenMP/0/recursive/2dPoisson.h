/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: 2dPoisson.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/applications/MGV/F95/OpenMP/0/recursive/2dPoisson.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: multigrid methode for 2D Poisson equation
 *******************************************************************/

#ifndef POISSON_H
#define POISSON_H


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>

#include "interface.h"

// Functions in Fortran
extern void fortran_entry_( int * level0, int * maxlevel0, int * outputform, int * v1, int * v2, double * w, double * L1, double * L2, double * time_for_MGV, double * omega, double * flop );
// --------------------

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
   myinttype output;
   myinttype outputform;
   /* additional parameters */
   myinttype maxsize;
} mydata_t;


#endif //POISSON_H


