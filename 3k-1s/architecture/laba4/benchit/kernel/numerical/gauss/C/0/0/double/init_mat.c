/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: init_mat.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/gauss/C/0/0/double/init_mat.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Gaussian Linear Equation System Solver
 *******************************************************************/

#include "gauss.h"
#include "interface.h"

int init_mat_c(FPT **A, int sa1, int sa2, FPT *b, int sb, FPT *x, int sx, int lbd1, int ubd1, int lbd2, int ubd2){
   int i,j;
   if(A==0 || b==0 || x==0) return (1);
   if(lbd1<0 || lbd2<0 ||  sa1<=ubd1 || sa2<=ubd2 || sb<=ubd1 || sx<=ubd2)return (2);
   
   for (i=lbd1; i<=ubd1; i++) {
      for (j=lbd2; j<=ubd2; j++) {
         A[i][j] = (i!=j)?((i!=j-4)?((i!=j+4)?0.0E0:18.0E0):18.0E0):3.0E0;
      }
   }
   
   for (i=lbd1; i<=ubd1; i++) {
      b[i] = (FPT)i;
   }
   
   for (j=lbd2; j<=ubd2; j++) {
      x[j] = 1.0E0;
   }

   return (0);
}

