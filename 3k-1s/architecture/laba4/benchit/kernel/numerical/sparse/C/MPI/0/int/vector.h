/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: vector.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/int/vector.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#ifndef VECTOR_H
#define VECTOR_H


#include "sparse.h"

extern DT * MPI_createVector( int size );
extern int * MPI_createVector_iDT( int size );
extern void MPI_copyVector( DT * source, DT * target, int size );
extern void MPI_copyVector_iDT( int * source, int * target, int size );
extern DT MPI_compareVector( DT * source, DT * target, int size );
extern void MPI_clearVector( DT * vec );
extern void MPI_printVector( DT * vec, int size );
extern void MPI_initRandomVector( DT * vec, int size );


#endif //VECTOR_H


