/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matmul1_funcs.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/F77/MPI/0/MPI_C/matmul1_funcs.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix multiply (Fortran 77) with MPI
 *******************************************************************/
 
double gettimeroverhead( void );
void deallocate( fds *pmem );
void allocateANDtouch( fds *pmem, int *pisize );
void entry_( void *ptr, int *pisize, int iunrolled );
double getseqentryoverhead( void *pmem );

extern void tstcas_ ( int *in, int *im, int *iunrolled, double *pda, double *pdb, double *pdc );

