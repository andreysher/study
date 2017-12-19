/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: reduce.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/reduction/C/MPI/0/reduce/reduce.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare the flops for different MPI reduction methodes
 *******************************************************************/

#ifndef __reduce_h
#define __reduce_h

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
typedef struct mydata {
   myinttype commsize;
   myinttype commrank;
   /* additional parameters */
   myinttype maxsize;
} mydata_t;

typedef struct {
   float x1, x2, x3;
} Vector;

typedef struct {
   float a11, a12, a21, a22;
} Matrix;

#include <mpi.h>

extern void reduce(int, int, void *, void *, int, MPI_Datatype *,
   MPI_Op *, double *);

extern void vecadd(void *, void *, int *, MPI_Datatype *);
extern void matmul(void *, void *, int *, MPI_Datatype *);

#endif


