/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: dotproduct.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/dotproduct/C/PThread/0/double/dotproduct.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for dot product of two vectors with posix threads
 *******************************************************************/

#include "dotproduct.h"

void *thread_func(void *mcb) {
   int i;
   double **mem = (double **)(mcb);
   double *a = mem[0], *b = mem[1], c;
   double count = *mem[2];

   IDL(3, printf("count: %d", (int)count));

   c = 0.0;
   for (i = 0; i < count; i++) {
      c += a[i] * b[i];
   }

   if (c != count) {
      printf("\n");
      printf("-------------------------------\n");
      printf("kernel error: calculation error\n");
      printf("expected %.1f and got %.1f\n", count, c);
      printf("-------------------------------\n");
      fflush(stdout);
      exit(127);
   }

   pthread_exit(NULL);

   return NULL;
}

