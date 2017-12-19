/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: barrier.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/C/PThread/0/n_mallocs/barrier.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension, for a given number of posix threads,
 *         mallocs in bi_entry addicted to actual problemSize
 *******************************************************************/

#include "barrier.h"

/* Initialization sequence of a barrier */
myinttype barrier_init(barrier_t * barrier, myinttype count) {
   myinttype status;
   barrier->threshold = barrier->counter = count;
   barrier->cycle = 0;
   status = pthread_mutex_init(&barrier->mutex, NULL);
   if (status != 0)
      return status;
   status = pthread_cond_init(&barrier->cv, NULL);
   if (status != 0) {
      pthread_mutex_destroy(&barrier->mutex);
      return status;
   }
   barrier->valid = BARRIER_VALID;
   return 0;
}

/* Destruction sequence of a barrier */
myinttype barrier_destroy(barrier_t * barrier) {
   myinttype status, status2;
   if (barrier->valid != BARRIER_VALID)
      return EINVAL;                   /* barrier to destroy must be valid */
   status = pthread_mutex_lock(&barrier->mutex);
   if (status != 0)
      return status;
   /* ensure no threads are waiting on the barrier */
   if (barrier->counter != barrier->threshold) {
      pthread_mutex_unlock(&barrier->mutex);
      return EBUSY;
   }
   barrier->valid = 0;                 /* set barrier invalid */
   status = pthread_mutex_unlock(&barrier->mutex);
   if (status != 0)
      return status;
   status = pthread_mutex_destroy(&barrier->mutex);
   status2 = pthread_cond_destroy(&barrier->cv);
   return (status != 0 ? status : status2);
}

/* Wait sequence on the barrier */
myinttype barrier_wait(barrier_t * barrier) {
   myinttype status, cancel, cycle;
   if (barrier->valid != BARRIER_VALID)
      return EINVAL;                   /* don't wait on invalid barrier */
   status = pthread_mutex_lock(&barrier->mutex);
   if (status != 0)
      return status;
   cycle = barrier->cycle;             /* store cycle value of barrier */
   /* decrease counter of barrier and check if zero */
   if (--barrier->counter == 0) {
      /* all threads wait -> release barrier */
      barrier->cycle++;                /* increase barrier's cycle counter */
      barrier->counter = barrier->threshold;    /* reset barrier's counter */
      status = pthread_cond_broadcast(&barrier->cv);    /* inform all threads
                                                         * waiting on the
                                                         * barrier's cond.var. */
      if (status == 0)
         status = -1;                  /* this allows to figure out which
                                        * thread droped the barrier */
   } else {
      /* some threads are still not waiting on the barrier */
      pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cancel);  /* prevent
                                                                 * getting
                                                                 * canceled */
      /* compare stored cycle value to cycle value of barrier */
      while (cycle == barrier->cycle) {
         /* this makes it possible to reuse the barrier as many times as
          * pleased */
         /* wait on the barrier's condition variable */
         status = pthread_cond_wait(&barrier->cv, &barrier->mutex);
         if (status != 0)
            break;
      }
   }
   pthread_mutex_unlock(&barrier->mutex);
   return status;
}

