/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/vecadd/F77/0/0/BjxxEBjxxPAi/work.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: <description>
 *******************************************************************/

#ifndef __work_h
#define __work_h

/*prototypes for work.c are in interface.h*/
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>

#include "interface.h"

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

/* structure saves both vectors that are added
 * result is saved in pdb
 ***/
typedef struct mydata {
   double *pda, *pdb;
   int repetitions;
   double sum;
} mydata_t;

extern void vecadd_(int *in, int *im, int *iunrolled, double *pda,
                    double *pdb);

#endif

