/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: mpi_functions.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/F77/MPI/0/MPI_F77/mpi_functions.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix multiply (Fortran 77) with MPI
 *******************************************************************/

#include "matmul.h"

#ifndef BENCHIT_MPI_FUNCTIONS_F77_H
#define BENCHIT_MPI_FUNCTIONS_F77_H

/* some Fortran compilers export symbols in s special way:
 * all letter are big letters
 */
#if (defined (_CRAY)    || \
     defined (_SR8000)  || \
     defined (_USE_OLD_STYLE_CRAY_TYPE))
#define mpivariables_  	        MPIVARIABLES
#define mpibinarybroadcast_	MPIBINARYBROADCAST
#define mpimatrixgather_	MPIMATRIXGATHER
#define mpimatrixscatter_	MPIMATRIXSCATTER
#define mpishiftmatrix_         MPISHIFTMATRIX
#define mpimatrixbroadcast_     MPIMATRIXBROADCAST
#endif

void mpivariables_(void);
void mpibinarybroadcast_( char *p, myinttype *bytes);
void mpimatrixbroadcast_( double *n, myinttype *rows, myinttype *cols);
void mpimatrixscatter_( double *a, double *b, myinttype *sizeall, 
			myinttype *sizeone);
void mpimatrixgather_( double *a, double *b, myinttype *sizeall, 
		       myinttype *sizeone);
void mpishiftmatrix_( double *n, myinttype *rows, myinttype *cols);
myinttype childs_( myinttype *id, myinttype *size);

#endif /* BENCHIT_MPI_FUNCTIONS_F77_H */

