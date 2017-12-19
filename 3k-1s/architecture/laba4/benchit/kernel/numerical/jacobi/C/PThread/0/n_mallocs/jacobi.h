/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: jacobi.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/C/PThread/0/n_mallocs/jacobi.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension, for a given number of posix threads,
 *         mallocs in bi_entry addicted to actual problemsize
 *******************************************************************/

#ifndef __jacobi_h
#define __jacobi_h

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <pthread.h>

#include "interface.h"

#include "barrier.h"

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
   double *diffnormArray;
   void *thread;
   myinttype maxn;
   myinttype nxy;
   myinttype mits;
   myinttype mitsdone;
   myinttype converged;
   myinttype stripes;
   barrier_t barrier;
   /* additional parameters */
   myinttype maxsize;
   myinttype threadCount;
   myinttype cpuCount;
} mydata_t;

typedef struct thread_tag {
   pthread_t thread_id;
   myinttype number;
   mydata_t *mdp;
} thread_t;

extern void *jacobi_thread_routine_ji(void *arg);
extern void *jacobi_thread_routine_ij(void *arg);

extern void twodinit(mydata_t * mdp);

#endif

