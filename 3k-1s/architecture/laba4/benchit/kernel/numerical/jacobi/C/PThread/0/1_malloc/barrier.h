/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: barrier.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/C/PThread/0/1_malloc/barrier.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension, for a given number of posix threads,
 *         one malloc for biggest dimension
 *******************************************************************/

#ifndef __barrier_h
#define __barrier_h

#include <pthread.h>
#include "errors.h"

#if (defined (_CRAYMPP) || \
	defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

typedef struct barrier_tag {
   pthread_mutex_t mutex;              /* control access to barrier */
   pthread_cond_t cv;                  /* wait for barrier */
   myinttype valid;                    /* set when valid */
   myinttype threshold;                /* num of threads required */
   myinttype counter;                  /* current number of threads */
   myinttype cycle;                    /* count cycles */
} barrier_t;

#define BARRIER_VALID 1

#define BARRIER_INITIALIZER(cnt) \
	{PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, \
	BARRIER_VALID, cnt, cnt, 0}

extern myinttype barrier_init(barrier_t * barrier, myinttype count);
extern myinttype barrier_destroy(barrier_t * barrier);
extern myinttype barrier_wait(barrier_t * barrier);

#endif

