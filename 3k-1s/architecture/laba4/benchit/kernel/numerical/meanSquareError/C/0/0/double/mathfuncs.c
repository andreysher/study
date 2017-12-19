/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: mathfuncs.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/meanSquareError/C/0/0/double/mathfuncs.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Solve a linear mean square error problem
 *******************************************************************/

#include "mse.h"

/****f* mathfuncs.c/mathematics
 * SYNOPSIS
 * The following functions are "mathematical operations" that are
 * needed for the mean square error kernel.
 * They can be found in the file named
 * kernel/mse_c/mathfuncs.c
 ***/

/****f* mathfuncs.c/mathematics::signum
 * SYNOPSIS
 * int signum(double dx)
 * DESCRIPTION
 * This function returns the sign of "dx".
 ***/
int signum(double dx) {
   return (dx < 0) ? -1 : 1;
}

/****f* mathfuncs.c/mathematics::matmul
 * SYNOPSIS
 * double **matmul(double **ppdmatrix1, int ix1, int iy1,
 *                 double **ppdmatrix2, int ix2, int iy2, int ioffset)
 * DESCRIPTION
 * - function multiplies matrix1 with x1 lines and y1 columns and
 *   matrix2 with x2-offset lines and y2 columns
 * - the offset is needed to be able to compute with "child matrixes"
 * - the function returns a matrix with x2 lines and y2 columns
 *   the numbers that are not computed are copied from matrix2
 * - matrix1 and matrix2 are freed at the end of the function
 ***/
double **matmul(double **ppdmatrix1, int ix1, int iy1, double **ppdmatrix2,
                int ix2, int iy2, int ioffset) {
   double **ppdresult = NULL;
   int ii = 0, ij = 0, ik = 0;

   IDL(1, printf("reached function matmul\n"));

   ppdresult = create2Darray(ix2, iy2);

   if (iy1 != ix2 - ioffset) {
      printf("Dimension error!\n");
   }

   IDL(2,
       printf
       ("reached begin of the loop that copies finished elements to the result\n"));
   for (ii = 0; ii < ioffset; ii++) {
      for (ij = 0; ij < iy2; ij++) {
         ppdresult[ii][ij] = ppdmatrix2[ii][ij];
      }
      for (ij = 0; ij < ix2; ij++) {
         ppdresult[ij][ii] = ppdmatrix2[ij][ii];
      }
   }
   IDL(2,
       printf
       ("reached end of the loop that copies finished elements to the result\n"));

   IDL(2, printf("reached begin of the loop that multiplies matrixes"));
   for (ii = 0; ii < ix1; ii++) {
      for (ij = 0; ij < iy2 - ioffset; ij++) {
         for (ik = 0; ik < iy1; ik++) {
            ppdresult[ii + ioffset][ij + ioffset] =
               ppdresult[ii + ioffset][ij + ioffset] +
               ppdmatrix1[ii][ik] * ppdmatrix2[ik + ioffset][ij + ioffset];
         }
      }
   }
   IDL(2, printf("reached end of the loop that multiplies matrixes"));

   free2Darray(ppdmatrix1, ix1);
   free2Darray(ppdmatrix2, ix2);

   IDL(1, printf("completed function matmul\n"));
   return ppdresult;
}

/****f* mathfuncs.c/mathematics::createQ
 * SYNOPSIS
 * double **createQ(double **ppdvector, int ilength, int ipos)
 * DESCRIPTION
 * - function creates a q-matrix to vector that has the dimension
 *   ilength-ipos-1
 * - first element of the regarded vector is ppdvector[ipos][ipos]
 *   last element is ppdvector[ilength-1][ipos]
 ***/
double **createQ(double **ppdvector, int ilength, int ipos) {
   double **ppdq = NULL;
   double dd = 0.0, dabsc = 0.0;
   int ii = 0, ij = 0;

   IDL(1, printf("reached function createQ\n"));

   ppdq = create2Darray(ilength - ipos, ilength - ipos);

   for (ii = 1 + ipos; ii < ilength; ii++) {
      dabsc = dabsc + ppdvector[ii][ipos] * ppdvector[ii][ipos];
   }
   dd =
      -signum(ppdvector[ipos][ipos]) * sqrt(ppdvector[ipos][ipos] *
                                            ppdvector[ipos][ipos] + dabsc);
   ppdvector[ipos][ipos] = ppdvector[ipos][ipos] - dd;

   IDL(2,
       printf
       ("reached begin of the loop that creates the Q-matrix from a vector\n"));
   for (ii = ipos; ii < ilength; ii++) {
      for (ij = ipos; ij < ilength; ij++) {
         ppdq[ii - ipos][ij - ipos] =
            1 / (dd * ppdvector[ipos][ipos]) * ppdvector[ii][ipos] *
            ppdvector[ij][ipos];
         if (ii == ij)
            ppdq[ii - ipos][ij - ipos] = ppdq[ii - ipos][ij - ipos] + 1;
      }
   }
   IDL(2,
       printf
       ("reached end of the loop that creates the Q-matrix from a vector\n"));

   /* vector is the matrix used in the kernel so it should not be changed after 
    * that function */
   ppdvector[ipos][ipos] = ppdvector[ipos][ipos] + dd;

   IDL(1, printf("completed function createQ\n"));
   return ppdq;
}

/****f* mathfuncs.c/mathematics::solve
 * SYNOPSIS
 * double *solve(double **ppdmatrix, int ivars)
 * DESCRIPTION
 * - function solves the system of equations that is an upper
 *   triangular matrix stored in ppdmatrix
 * - the last column of ppdmatrix is regarded as the
 *   right side of the equations
 * - ppdmatrix is supposed to have ivars+1 columns
 ***/
double *solve(double **ppdmatrix, int ivars) {
   double *pdsolution = NULL;
   int ii = 0, ij = 0;

   IDL(1, printf("reached function solve\n"));

   pdsolution = calloc(ivars, sizeof(double));

   IDL(2, printf("reached begin of the loop that solves the triangle matrix"));
   for (ii = ivars - 1; ii > -1; ii--) {
      for (ij = ivars - 1; ij > ii; ij--) {
         pdsolution[ii] = pdsolution[ii] - ppdmatrix[ii][ij] * pdsolution[ij];
      }
      pdsolution[ii] =
         (pdsolution[ii] + ppdmatrix[ii][ivars]) / ppdmatrix[ii][ii];
   }
   IDL(2, printf("reached begin of the loop that solves the triangle matrix"));

   IDL(1, printf("completed function solve\n"));
   return pdsolution;
}

