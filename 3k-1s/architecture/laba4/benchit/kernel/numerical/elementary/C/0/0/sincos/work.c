/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/elementary/C/0/0/sincos/work.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Execution time of mathematical operations SINUS / COSINUS
 *         addict to input value
 *******************************************************************/

#include "work.h"
#include "interface.h"

void entry_(myinttype *size) {
   FPT x, y;
   register FPT *zx=&x, *zy=&y;
   // register int a;
   x=(FPT)(*size);
   /* 1000 * 1000 macht 1 Mio Operationen */
   // for(a=0; a<1000; a++) {
   THOUSAND(zx,zy)
   // }
}

void mathopsin_(FPT *x, FPT *y) {
   if(*x!=0)
      *y=sin(*x);
}

void mathopcos_(FPT *x, FPT *y) {
   if(*x!=0)
      *y=cos(*x);
}

