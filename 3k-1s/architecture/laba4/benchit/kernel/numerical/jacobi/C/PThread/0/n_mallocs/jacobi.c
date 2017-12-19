/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: jacobi.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/C/PThread/0/n_mallocs/jacobi.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension, for a given number of posix threads,
 *         mallocs in bi_entry addicted to actual problemSize
 *******************************************************************/

#include "jacobi.h"

void sweep2da_ji(mydata_t * mdp, myinttype is, myinttype ie, myinttype js,
                 myinttype je);
void sweep2da_ij(mydata_t * mdp, myinttype is, myinttype ie, myinttype js,
                 myinttype je);
void sweep2db_ji(mydata_t * mdp, myinttype is, myinttype ie, myinttype js,
                 myinttype je);
void sweep2db_ij(mydata_t * mdp, myinttype is, myinttype ie, myinttype js,
                 myinttype je);
void diff2d_ji(mydata_t * mdp, myinttype id, myinttype is, myinttype ie,
               myinttype js, myinttype je);
void diff2d_ij(mydata_t * mdp, myinttype id, myinttype is, myinttype ie,
               myinttype js, myinttype je);

/* specifies the work each thread has to do */
void *jacobi_thread_routine_ji(void *arg) {
   thread_t *self = (thread_t *) arg;
   myinttype i = 0, it = 0, status = 0;
   myinttype is = 0, js = 0;

   is = (self->number % self->mdp->stripes) * self->mdp->nxy + 1;
   js =
      ((myinttype) floor((self->number * 1.0) / self->mdp->stripes)) *
      self->mdp->nxy + 1;

   for (it = 0; it < self->mdp->mits; it++) {
      if (self->mdp->converged == 1)
         break;
      /* ################ */
      /* ### step 1 ##### */
      /* ################ */
      sweep2db_ji(self->mdp, is, is + self->mdp->nxy - 1, js,
                  js + self->mdp->nxy - 1);
      status = barrier_wait(&self->mdp->barrier);
      if (status > 0)
         err_abort(status, "Wait on barrier");
      /* ################ */
      /* ### step 2 ##### */
      /* ################ */
      sweep2da_ji(self->mdp, is, is + self->mdp->nxy - 1, js,
                  js + self->mdp->nxy - 1);
      status = barrier_wait(&self->mdp->barrier);
      if (status > 0)
         err_abort(status, "Wait on barrier");
      /* ################ */
      /* ### step 3 ##### */
      /* ################ */
      diff2d_ji(self->mdp, self->number, is, is + self->mdp->nxy - 1, js,
                js + self->mdp->nxy - 1);
      status = barrier_wait(&self->mdp->barrier);
      if (status > 0)
         err_abort(status, "Wait on barrier");
      if (self->number == (self->mdp->threadCount - 1)) {
         self->mdp->diffnorm = 0.0;
         for (i = 0; i < self->mdp->threadCount; i++) {
            self->mdp->diffnorm += self->mdp->diffnormArray[i];
            self->mdp->diffnormArray[i] = 0.0;
         }
         if (self->mdp->diffnorm < 1.3e-3) {
            self->mdp->mitsdone = it;
            self->mdp->converged = 1;
         }
      }
      status = barrier_wait(&self->mdp->barrier);
      if (status > 0)
         err_abort(status, "Wait on barrier");
   }

   if (self->number == (self->mdp->threadCount - 1)) {
      if (self->mdp->diffnorm >= 1.3e-3) {
         self->mdp->mitsdone = it;
      }
   }

   pthread_exit(NULL);

   return NULL;
}

/* specifies the work each thread has to do */
void *jacobi_thread_routine_ij(void *arg) {
   thread_t *self = (thread_t *) arg;
   myinttype it = 0, status = 0;
   myinttype is = 0, js = 0;
   myinttype i = 0;

   is = (self->number % self->mdp->stripes) * self->mdp->nxy + 1;
   js =
      ((myinttype) floor((self->number * 1.0) / self->mdp->stripes)) *
      self->mdp->nxy + 1;

   for (it = 0; it < self->mdp->mits; it++) {
      if (self->mdp->converged == 1)
         break;
      /* ################ */
      /* ### step 1 ##### */
      /* ################ */
      sweep2db_ij(self->mdp, is, is + self->mdp->nxy - 1, js,
                  js + self->mdp->nxy - 1);
      status = barrier_wait(&self->mdp->barrier);
      if (status > 0)
         err_abort(status, "Wait on barrier");
      /* ################ */
      /* ### step 2 ##### */
      /* ################ */
      sweep2da_ij(self->mdp, is, is + self->mdp->nxy - 1, js,
                  js + self->mdp->nxy - 1);
      status = barrier_wait(&self->mdp->barrier);
      if (status > 0)
         err_abort(status, "Wait on barrier");
      /* ################ */
      /* ### step 3 ##### */
      /* ################ */
      diff2d_ij(self->mdp, self->number, is, is + self->mdp->nxy - 1, js,
                js + self->mdp->nxy - 1);
      status = barrier_wait(&self->mdp->barrier);
      if (status > 0)
         err_abort(status, "Wait on barrier");
      if (self->number == (self->mdp->threadCount - 1)) {
         self->mdp->diffnorm = 0.0;
         for (i = 0; i < self->mdp->threadCount; i++) {
            self->mdp->diffnorm += self->mdp->diffnormArray[i];
            self->mdp->diffnormArray[i] = 0.0;
         }
         if (self->mdp->diffnorm < 1.3e-3) {
            self->mdp->mitsdone = it;
            self->mdp->converged = 1;
         }
      }
      status = barrier_wait(&self->mdp->barrier);
      if (status > 0)
         err_abort(status, "Wait on barrier");
   }

   if (self->number == (self->mdp->threadCount - 1)) {
      if (self->mdp->diffnorm >= 1.3e-3) {
         self->mdp->mitsdone = it;
      }
   }

   pthread_exit(NULL);

   return NULL;
}

void twodinit(mydata_t * mdp) {
   /* already zeroed at memory allocation */
   myinttype i = 0, j = 0, index = 0;
   mdp->converged = 0;
   mdp->diffnorm = 0.0;
   mdp->mitsdone = 0;

   for (i = 0; i < mdp->threadCount; i++) {
      mdp->diffnormArray[i] = 0.0;
   }
   for (j = 0; j < mdp->maxn; j++) {
      for (i = 0; i < mdp->maxn; i++) {
         index = mdp->maxn * j + i;
         mdp->a[index] = 0.0;
         mdp->b[index] = 0.0;
         mdp->f[index] = 0.0;
      }
   }
   for (i = 1; i < (mdp->maxn - 1); i++) {
      index = i;
      mdp->a[index] = 1.0;
      mdp->b[index] = 1.0;
   }
   for (j = 1; j < (mdp->maxn - 1); j++) {
      index = mdp->maxn * j;
      mdp->a[index] = 1.0;
      mdp->b[index] = 1.0;
   }
}

void sweep2da_ji(mydata_t * mdp, myinttype is, myinttype ie, myinttype js,
                 myinttype je) {
   myinttype index = 0, i = 0, j = 0;

   for (j = js; j <= je; j++) {
      for (i = is; i <= ie; i++) {
         index = j * mdp->maxn + i;
         mdp->b[index] =
            0.25 * (mdp->a[index - mdp->maxn] + mdp->a[index - 1] +
                    mdp->a[index + mdp->maxn] + mdp->a[index + 1]) -
            mdp->h * mdp->h * mdp->f[index];
      }
   }
}
void sweep2da_ij(mydata_t * mdp, myinttype is, myinttype ie, myinttype js,
                 myinttype je) {
   myinttype index = 0, i = 0, j = 0;

   for (i = is; i <= ie; i++) {
      for (j = js; j <= je; j++) {
         index = j * mdp->maxn + i;
         mdp->b[index] =
            0.25 * (mdp->a[index - mdp->maxn] + mdp->a[index - 1] +
                    mdp->a[index + mdp->maxn] + mdp->a[index + 1]) -
            mdp->h * mdp->h * mdp->f[index];
      }
   }
}

void sweep2db_ji(mydata_t * mdp, myinttype is, myinttype ie, myinttype js,
                 myinttype je) {
   myinttype index = 0, i = 0, j = 0;

   for (j = js; j <= je; j++) {
      for (i = is; i <= ie; i++) {
         index = j * mdp->maxn + i;
         mdp->a[index] =
            0.25 * (mdp->b[index - mdp->maxn] + mdp->b[index - 1] +
                    mdp->b[index + mdp->maxn] + mdp->b[index + 1]) -
            mdp->h * mdp->h * mdp->f[index];
      }
   }
}
void sweep2db_ij(mydata_t * mdp, myinttype is, myinttype ie, myinttype js,
                 myinttype je) {
   myinttype index = 0, i = 0, j = 0;

   for (i = is; i <= ie; i++) {
      for (j = js; j <= je; j++) {
         index = j * mdp->maxn + i;
         mdp->a[index] =
            0.25 * (mdp->b[index - mdp->maxn] + mdp->b[index - 1] +
                    mdp->b[index + mdp->maxn] + mdp->b[index + 1]) -
            mdp->h * mdp->h * mdp->f[index];
      }
   }
}

void diff2d_ji(mydata_t * mdp, myinttype id, myinttype is, myinttype ie,
               myinttype js, myinttype je) {
   myinttype index = 0, i = 0, j = 0;
   double diff = 0.0, sum = 0.0;

   for (j = js; j <= je; j++) {
      for (i = is; i <= ie; i++) {
         index = j * mdp->maxn + i;
         diff = mdp->a[index] - mdp->b[index];
         sum += diff * diff;
      }
   }
   mdp->diffnormArray[id] += sum;
}

void diff2d_ij(mydata_t * mdp, myinttype id, myinttype is, myinttype ie,
               myinttype js, myinttype je) {
   myinttype index = 0, i = 0, j = 0;
   double diff = 0.0, sum = 0.0;

   for (i = is; i <= ie; i++) {
      for (j = js; j <= je; j++) {
         index = j * mdp->maxn + i;
         diff = mdp->a[index] - mdp->b[index];
         sum += diff * diff;
      }
   }
   mdp->diffnormArray[id] += sum;
}

