/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: dgemm.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/OpenMP/MKL/double/dgemm.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix Multiply, BLAS, MKL (C) - OpenMP version
 *******************************************************************/

#ifndef DGEMM__

#define DGEMM__
	typedef struct floating_data_struct {
	  double *feld1, *feld2, *feld3;
	} fds;

//	long MIN, MAX, INCREMENT; 

#else
//	extern long MIN, MAX, INCREMENT; 

#endif


