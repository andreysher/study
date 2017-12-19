/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matrix.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/float/matrix.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#ifndef MATRIX_H
#define MATRIX_H


#include "sparse.h"
#include "vector.h"

extern DT * MPI_MatxVec( DT ** matrix, int m, int n, DT * x, int sizeOfX );
extern DT * MPI_MatxVec_pruef( DT ** matrix, int m, int n, DT * x, int sizeOfX );

extern DT** MPI_createMatrix( int m, int n );
extern void MPI_clearMatrix( DT ** matrix );
extern void MPI_initRandomMatrix( DT ** matrix, int m, int n, float percent );
extern void MPI_initZERO( DT ** matrix, int m, int n );
extern void MPI_initIDENTITY( DT ** matrix, int m, int n );
extern void MPI_initDIAG( DT ** matrix, int m, int n, int diag );
extern void MPI_init5PSTAR( DT ** matrix, int m, int n );
extern void MPI_printMatrix( DT ** matrix, int m, int n );


#endif //MATRIX_H


