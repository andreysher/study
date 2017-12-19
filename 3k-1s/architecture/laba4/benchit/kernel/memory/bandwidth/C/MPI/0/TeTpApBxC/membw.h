/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: membw.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/bandwidth/C/MPI/0/TeTpApBxC/membw.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Memory Bandwidth (C)
 *******************************************************************/

#ifndef BENCHIT_MEMORY_TEST_H
#define BENCHIT_MEMORY_TEST_H

typedef struct vs{
  double *a;
  double *b;
  double *c;
} vec_struct;

double mem_read( double *a, double *b, double *c,int problemsize);

#endif /* BENCHIT_MEMORY_TEST_H */


