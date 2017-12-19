/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: memread.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/bandwidth/C/MPI/0/TeTpApBxC/memread.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Memory Bandwidth (C)
 *******************************************************************/

#include "interface.h"
#include "membw.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

double mem_read(double *a, double *b, double *c, int problemSize) {
  long i;
  double temp=0.0;

  if(problemSize==0)
    return temp;	


  for(i=0; i<problemSize; i++) {
    temp+=a[i]+b[i]*c[i];
      }
  return temp;
}


