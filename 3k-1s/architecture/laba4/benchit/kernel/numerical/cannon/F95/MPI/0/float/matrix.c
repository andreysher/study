/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matrix.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/cannon/F95/MPI/0/float/matrix.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: a MPI version of matrix-matrix multiplication
 *         (cannon algotithm)
 *******************************************************************/

#include "matrix.h"
#include "mpicomm.h"

/*
* deliver max{ i1, i2 }
*/
int iMax(int i1, int i2) {
   if (i1 > i2) {
      return i1;
   } else {
      return i2;
   }
}

/*
* deliver min{ i1, i2 }
*/
int iMin(int i1, int i2) {
   if (i1 < i2) {
      return i1;
   } else {
      return i2;
   }
}

/*
* create an empty part of matrix for each of the n processors
* processor 1 to n-1 get the same number of rows of the matrix
* the last processor get a number of rows less than for the others 
*/
DT *createMatrix(int m, int n) {
   int i;
   DT *matrix;

   IDL(INFO, printf("\n---->Entered MPI_createMatrix() for rank=%i\n", rank));

   matrix = (DT *) malloc(m * n * sizeof(DT));
   assert(matrix != NULL);

   IDL(INFO, printf("\n<----Exit MPI_createMatrix() for rank=%i\n", rank));

   return matrix;
}

/*
* deallocate the memory for the matrix
*/
void clearMatrix(DT * matrix) {
   IDL(INFO, printf("\n---->Entered MPI_clearMatrix() for rank=%i\n", rank));

   free(matrix);

   IDL(INFO, printf("\n<----Exit MPI_clearMatrix() for rank=%i\n", rank));
}

/*
* fill the matrix with 0
*/
void initZERO(DT * matrix, int m, int n) {
   int i, j;

   IDL(INFO, printf("\n---->Entered initZERO() for rank=%i\n", rank));

   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
         matrix[i * n + j] = 0;
      }
   }

   IDL(INFO, printf("<----Exit initZERO() for rank=%i\n", rank));
}

/*
* fill the matrix with random numbers
*	x in [1, 2, ... , 9] for long
*	x in [1.0, ..., 9.0] for double
*/
void initRandomMatrix(DT * matrix, int m, int n) {
   int i, j;

   IDL(INFO, printf("\n---->Entered initRandomMatrix() for rank=%i\n", rank));

   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
         matrix[i * n + j] = 1 + (DT) (9.0 * rand() / (RAND_MAX + 1.0));
      }
   }

   IDL(INFO, printf("<----Exit initRandomMatrix() for rank=%i\n", rank));
}

/*
* make a multiplication of the matrixparts of A and B on the processor
* after the multiplication there will be the next shift to get the next parts from the neighbour in the grid
*/
DT **MatxMat(DT * matrixS, int *mS, int *nS, DT * matrixA, int *mA, int *nA,
             DT * matrixB, int *mB, int *nB, GridPosition gridpos, int cpu_x,
             int cpu_y, void (*ptr_to_perm) (DT * matrixS, DT * matrixA,
                                             DT * matrixB, int *mA, int *nA,
                                             int *nB)) {
   int c = 0;
   DT **pointer = NULL;

   IDL(INFO, printf("\n---->Entered MatxMat() for rank=%i\n", rank));

   for (c = 0; c < cpu_x; c++) {
      IDL(INFO, printf("\nrank=%i  step=%i\n", rank, c + 1));
      IDL(INFO,
          printf("\n*mS,*nS==%i,%i  *mA,*nA==%i,%i  *mB,*nB==%i,%i\n", *mS, *nS,
                 *mA, *nA, *mB, *nB));
      // assert(*mS==*mA && *nS==*nB && *nA==*mB);

      (*ptr_to_perm) (matrixS, matrixA, matrixB, mA, nA, nB);

      pointer =
         nextShift(matrixA, mA, nA, matrixB, mB, nB, gridpos, cpu_x, cpu_y);
      if (pointer[0] != NULL)
         matrixA = pointer[0];
      if (pointer[1] != NULL)
         matrixB = pointer[1];
      free(pointer);

      IDL(INFO, printf("nextMatrixA in step=%i:\n", c + 1));
      IDL(INFO, printMatrix(matrixA, *mA, *nA));
      IDL(INFO, printf("nextMatrixB in step=%i:\n", c + 1));
      IDL(INFO, printMatrix(matrixB, *mB, *nB));
   }

   IDL(INFO, printf("<----Exit MatxMat() for rank=%i\n", rank));

   pointer = (DT **) malloc(2 * sizeof(DT *));
   pointer[0] = matrixA;
   pointer[1] = matrixB;

   return pointer;
}

void printMatrix(DT * matrix, int m, int n) {
   int i, j;

   IDL(INFO, printf("\n---->Entered printMatrix() for rank=%i\n", rank));

   for (i = 0; i < m; i++) {
      for (j = 0; j < n - 1; j++) {
         printf(FORMAT1, matrix[i * n + j]);
      }
      printf(FORMAT2, matrix[i * n + (n - 1)]);
   }

   IDL(INFO, printf("<----Exit printMatrix() for rank=%i\n", rank));
}

