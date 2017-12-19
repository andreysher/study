/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: jacobi.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/C/0/0/n_mallocs/jacobi.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension, mallocs in bi_entry addicted to actual
 *         problemsize
 *******************************************************************/

#ifndef __jacobi_h
#define __jacobi_h

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
typedef struct mydata {
   double *a;
   double *b;
   double *f;
   double h;
   double diffnorm;
   myinttype maxn;
   myinttype mits;
   myinttype mitsdone;
} mydata_t;

extern void jacobi_routine_ji(mydata_t *);
extern void jacobi_routine_ij(mydata_t *);

extern void twodinit(mydata_t *);

#endif

