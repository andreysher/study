/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/SSE2_Intrinsics/0/aligned_m128d/work.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: SSE2 Matrix Multiply (C), aligned data
 *******************************************************************/

#ifndef __work_h
#define __work_h
#endif

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
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
/*
   <variabletype1> <variablename1>;
   <variabletype2> <variablename2>;
*/
} mydata_t;

void multasseijk_( double *a, double *b, double *c, int *size );
void multasseikj_( double *a, double *b, double *c, int *size );
void multassejik_( double *a, double *b, double *c, int *size );
void multassejki_( double *a, double *b, double *c, int *size );
void multassekji_( double *a, double *b, double *c, int *size );
void multassekij_( double *a, double *b, double *c, int *size );
void multassealignijk_( double *a, double *b, double *c, int *size );
void multassealignikj_( double *a, double *b, double *c, int *size );
void multassealignjik_( double *a, double *b, double *c, int *size );
void multassealignjki_( double *a, double *b, double *c, int *size );
void multassealignkji_( double *a, double *b, double *c, int *size );
void multassealignkij_( double *a, double *b, double *c, int *size );


