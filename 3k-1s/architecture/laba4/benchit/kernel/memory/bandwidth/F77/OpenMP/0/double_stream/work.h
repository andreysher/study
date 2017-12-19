/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/bandwidth/F77/OpenMP/0/double_stream/work.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measure Bandwidth inspired by STREAM benchmark
 *         (FORTRAN OMP-version)
 *
 *         according to the rules, reffer this Benchmark as:
 *         "BenchIT kernel based on a variant of the STREAM benchmark code"
 *         when publishing results
 *
 *         This file is a header for the work-part of the kernel (work.f) 
 *******************************************************************/

#ifndef __work_h
#define __work_h
#endif

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata
{
   double* a;
   double* b;
   double* c;
} mydata_t;

extern void copy_( double *a, double *b, int *size);
extern void scale_( double *a, double *b, double *scalar, int *size);
extern void add_( double *a, double *b, double *c, int *size);
extern void triad_( double *a, double *b, double *c, double *scalar, int *size);


