/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: mpicomm.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/cannon/F95/MPI/0/double/mpicomm.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: a MPI version of matrix-matrix multiplication
 *         (cannon algotithm)
 *******************************************************************/

#ifndef MPICOMM_H
#define MPICOMM_H

#include "cannon.h"
#include "matrix.h"

extern void createMyMPIComm(int globalsize, int m, int n);
extern void freeMyMPIComm();
extern DT **initialShift(DT * matrixA, int *mA, int *nA, DT * matrixB, int *mB,
                         int *nB, GridPosition gridpos, int cpu_x, int cpu_y);
extern DT **nextShift(DT * matrixA, int *mA, int *nA, DT * matrixB, int *mB,
                      int *nB, GridPosition gridpos, int cpu_x, int cpu_y);
extern DT **undoShift(DT * matrixA, int *mA, int *nA, DT * matrixB, int *mB,
                      int *nB, GridPosition gridpos, int cpu_x, int cpu_y);

#endif                                 // MPICOMM_H

