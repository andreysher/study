/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: vector.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/0/0/double/vector.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "sparse.h"

extern DT * createVector( long size );
extern void copyVector( DT * source, DT * target, long size );
extern DT compareVector( DT * source, DT * target, long size );
extern void clearVector( DT * vec );
extern void printVector( DT * vec, long size );
extern void initRandomVector( DT * vec, long size );


