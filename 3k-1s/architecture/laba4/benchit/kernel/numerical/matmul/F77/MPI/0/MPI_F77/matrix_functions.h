/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matrix_functions.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/F77/MPI/0/MPI_F77/matrix_functions.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix multiply (Fortran 77) with MPI
 *******************************************************************/
 
#include "matmul.h"

#ifndef BENCHIT_HAVE_MATRIX_FUNCTIONS_H
#define BENCHIT_HAVE_MATRIX_FUNCTIONS_H

void matrixinit_( double *a, double *b, myinttype *sizeall, 
		  myinttype *sizeone);
void matrixprint_( double *a, myinttype *rows, myinttype *cols);
void matrixfill_(double *a, myinttype *rows, myinttype *cols, double *value);
void matmul_(double *a, double *b, double *c, myinttype *rowsa, 
	     myinttype *colsa, myinttype *rowsb, myinttype *colsb);

#endif /* BENCHIT_HAVE_MATRIX_FUNCTIONS_H */


