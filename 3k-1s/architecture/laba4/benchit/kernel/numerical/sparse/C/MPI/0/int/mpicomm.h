/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: mpicomm.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/int/mpicomm.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#ifndef MPICOMM_H
#define MPICOMM_H


#include "sparse.h"

extern void createMyMPIComm( int m );
extern void freeMyMPIComm();


#endif //MPICOMM_H


