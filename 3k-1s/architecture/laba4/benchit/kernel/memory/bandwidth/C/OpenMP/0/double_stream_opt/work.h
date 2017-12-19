/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/bandwidth/C/OpenMP/0/double_stream_opt/work.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measure Bandwidth inspired by STREAM benchmark (C OMP-version)
 *
 * according to the rules, reffer this Benchmark as:
 * "BenchIT kernel based on a variant of the STREAM benchmark code"
 * when publishing results
 *
 *******************************************************************/
 
#include "interface.h"
#include <omp.h>

/* for sched_setaffinity */
//#define _GNU_SOURCE
//#include <sched.h>

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata
{
   double** a;
   double** b;
   double** c;
} mydata_t;

extern double copy_( double **a, double **b, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads);
extern double scale_( double **a, double **b, double scalar, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads);
extern double add_( double **a, double **b, double **c, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads);
extern double triad_( double **a, double **b, double **c, double scalar, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads);
