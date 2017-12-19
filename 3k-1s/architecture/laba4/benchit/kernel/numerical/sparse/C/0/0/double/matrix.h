/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matrix.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/0/0/double/matrix.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "sparse.h"

extern DT * MatxVec( DT ** matrix, long m, long n, DT * vec, long sizeOfVec );

extern DT** createMatrix( long m, long n );
extern void clearMatrix( DT ** matrix );
extern void initRandomMatrix( DT ** matrix, long m, long n, double percent );
extern void initZERO( DT ** matrix, long m, long n );
extern void initIDENTITY( DT ** matrix, long m, long n );
extern void initDIAG( DT ** matrix, long m, long n, long diag );
extern void init5PSTAR( DT ** matrix, long m, long n );
extern void printMatrix( DT ** matrix, long m, long n );


