/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matrix.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/cannon/F95/MPI/0/int/matrix.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: a MPI version of matrix-matrix multiplication
 *         (cannon algotithm)
 *******************************************************************/

#ifndef MATRIX_H
#define MATRIX_H

#include "cannon.h"

extern DT **MatxMat(DT * matrixS, int *mS, int *nS, DT * matrixA, int *mA,
                    int *nA, DT * matrixB, int *mB, int *nB,
                    GridPosition gridpos, int cpu_x, int cpu_y,
                    void (*ptr_to_perm) (DT * matrixS, DT * matrixA,
                                         DT * matrixB, int *mA, int *nA, int *nB));

/*
extern DT ** pruef_MatxMat( int mA, int nA, int mB, int nB, DT ** matrixA, int _mA, int _nA, DT ** matrixB, int _mB, int _nB, GridPosition gridpos, int cpu_x, int cpu_y );
extern DT ** gatherMatrix( int m, int n, DT ** matrix, int _m, int _n, GridPosition gridpos, int cpu_x, int cpu_y );
*/

extern DT *createMatrix(int m, int n);
extern void clearMatrix(DT * matrix);
extern void printMatrix(DT * matrix, int m, int n);
extern void initZERO(DT * matrix, int m, int n);
extern void initRandomMatrix(DT * matrix, int m, int n);

#endif                                 // MATRIX_H

