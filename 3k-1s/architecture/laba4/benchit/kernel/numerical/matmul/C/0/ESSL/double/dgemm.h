/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: dgemm.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/0/ESSL/double/dgemm.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix Multiply, BLAS, ESSL (C)
 *******************************************************************/

#ifndef DGEMM__

#define DGEMM__
	typedef struct floating_data_struct {
	  double *feld1, *feld2, *feld3;
	} fds;

//	long bi_dgemm_start, bi_dgemm_stop, bi_dgemm_increment; 

#else
//	extern long bi_dgemm_start, bi_dgemm_stop, bi_dgemm_increment; 

#endif


