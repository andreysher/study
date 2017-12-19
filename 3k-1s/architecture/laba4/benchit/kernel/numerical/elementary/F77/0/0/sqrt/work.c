/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/elementary/F77/0/0/sqrt/work.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for Square Root addict to input value
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

