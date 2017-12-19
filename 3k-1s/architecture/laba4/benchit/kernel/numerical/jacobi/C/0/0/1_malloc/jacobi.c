/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: jacobi.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/C/0/0/1_malloc/jacobi.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension, one malloc for biggest dimension
 *******************************************************************/

#include "jacobi.h"

void printstat(double *, myinttype);
void sweep2da_ji(mydata_t *);
void sweep2da_ij(mydata_t *);
void sweep2db_ji(mydata_t *);
void sweep2db_ij(mydata_t *);
double diff2d_ji(mydata_t *);
double diff2d_ij(mydata_t *);

/* use this to print out one of the matrices */
void printstat(double *v, myinttype size) {
   myinttype i, j, index;
   for (j = 0; j < size; j++) {
      for (i = 0; i < size; i++) {
         index = j * size + i;
         printf("[%d:%d]=%1.3f ", i, j, v[index]);
      }
      printf("\n");
   }
   printf("\n");
}

/* specifies the work to do */
void jacobi_routine_ji(mydata_t * mdp) {
   myinttype it = 0;
   for (it = 0; it < mdp->mits; it++) {
      mdp->diffnorm = 0.0;
      sweep2db_ji(mdp);
      sweep2da_ji(mdp);
      mdp->diffnorm = diff2d_ji(mdp);
      if (mdp->diffnorm < 1.3e-3) {
         mdp->mitsdone = it;
         break;
      }
   }
   if (mdp->diffnorm >= 1.3e-3) {
//      printf("failed to converge!\n\n");
      mdp->mitsdone = it;
   }
}

/* specifies the work to do */
void jacobi_routine_ij(mydata_t * mdp) {
   myinttype it = 0;
   for (it = 0; it < mdp->mits; it++) {
      mdp->diffnorm = 0.0;
      sweep2db_ij(mdp);
      sweep2da_ij(mdp);
      mdp->diffnorm = diff2d_ij(mdp);
      if (mdp->diffnorm < 1.3e-3) {
         mdp->mitsdone = it;
         break;
      }
   }
   if (mdp->diffnorm >= 1.3e-3) {
//      printf("failed to converge!\n\n");
      mdp->mitsdone = it;
   }
}

void twodinit(mydata_t * mdp) {
   myinttype i = 0, j = 0, index = 0;
   mdp->diffnorm = 0.0;
   mdp->mitsdone = 0;
   for (j = 0; j < mdp->maxn; j++) {
      for (i = 0; i < mdp->maxn; i++) {
         index = j * mdp->maxn + i;
         mdp->a[index] = 0.0;
         mdp->b[index] = 0.0;
         mdp->f[index] = 0.0;
      }
   };
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

void sweep2da_ji(mydata_t * mdp) {
   myinttype index = 0, i = 0, j = 0;
   for (j = 1; j < mdp->maxn - 1; j++) {
      for (i = 1; i < mdp->maxn - 1; i++) {
         index = j * mdp->maxn + i;
         mdp->b[index] =
            0.25 * (mdp->a[index - mdp->maxn] + mdp->a[index - 1] +
                    mdp->a[index + mdp->maxn] + mdp->a[index + 1]) -
            mdp->h * mdp->h * mdp->f[index];
      }
   }
}
void sweep2da_ij(mydata_t * mdp) {
   myinttype index = 0, i = 0, j = 0;
   for (i = 1; i < mdp->maxn - 1; i++) {
      for (j = 1; j < mdp->maxn - 1; j++) {
         index = j * mdp->maxn + i;
         mdp->b[index] =
            0.25 * (mdp->a[index - mdp->maxn] + mdp->a[index - 1] +
                    mdp->a[index + mdp->maxn] + mdp->a[index + 1]) -
            mdp->h * mdp->h * mdp->f[index];
      }
   }
}

void sweep2db_ji(mydata_t * mdp) {
   myinttype index = 0, i = 0, j = 0;
   for (j = 1; j < mdp->maxn - 1; j++) {
      for (i = 1; i < mdp->maxn - 1; i++) {
         index = j * mdp->maxn + i;
         mdp->a[index] =
            0.25 * (mdp->b[index - mdp->maxn] + mdp->b[index - 1] +
                    mdp->b[index + mdp->maxn] + mdp->b[index + 1]) -
            mdp->h * mdp->h * mdp->f[index];
      }
   }
}
void sweep2db_ij(mydata_t * mdp) {
   myinttype index = 0, i = 0, j = 0;
   for (i = 1; i < mdp->maxn - 1; i++) {
      for (j = 1; j < mdp->maxn - 1; j++) {
         index = j * mdp->maxn + i;
         mdp->a[index] =
            0.25 * (mdp->b[index - mdp->maxn] + mdp->b[index - 1] +
                    mdp->b[index + mdp->maxn] + mdp->b[index + 1]) -
            mdp->h * mdp->h * mdp->f[index];
      }
   }
}

double diff2d_ji(mydata_t * mdp) {
   myinttype index = 0, i = 0, j = 0;
   double diff = 0.0, sum = 0.0;
   for (j = 1; j < mdp->maxn - 1; j++) {
      for (i = 1; i < mdp->maxn - 1; i++) {
         index = j * mdp->maxn + i;
         diff = mdp->a[index] - mdp->b[index];
         sum += diff * diff;
      }
   }
   return sum;
}
double diff2d_ij(mydata_t * mdp) {
   myinttype index = 0, i = 0, j = 0;
   double diff = 0.0, sum = 0.0;
   for (i = 1; i < mdp->maxn - 1; i++) {
      for (j = 1; j < mdp->maxn - 1; j++) {
         index = j * mdp->maxn + i;
         diff = mdp->a[index] - mdp->b[index];
         sum += diff * diff;
      }
   }
   return sum;
}

