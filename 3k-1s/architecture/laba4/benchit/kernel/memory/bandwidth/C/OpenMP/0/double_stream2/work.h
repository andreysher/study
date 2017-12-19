
#ifndef __work_h
#define __work_h
#endif

typedef struct mydata
{
   double* a;
   double* b;
} mydata_t;

extern void copy_( double *a, double *b, int size);
extern void fill_( double *a, double q, int size);
extern void daxpy_( double *a, double q, double *b, int size);
extern double sum_( double *a, int size);
