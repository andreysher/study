
#include "interface.h"
#include <omp.h>

/* for sched_setaffinity */
//#define _GNU_SOURCE
//#include <sched.h>

typedef struct mydata
{
   double** a;
   double** b;
} mydata_t;

extern double copy_( double **a, double **b, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads);
extern double sum_( double **a, double *result, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads);
extern double fill_( double **a, double scalar, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads);
extern double daxpy_( double **a, double **b, double scalar, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads);
