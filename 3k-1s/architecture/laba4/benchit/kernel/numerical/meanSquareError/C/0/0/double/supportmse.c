/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: supportmse.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/meanSquareError/C/0/0/double/supportmse.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Solve a linear mean square error problem
 *******************************************************************/
 
#include "mse.h"

/****f* supportmse.c/support
 * SYNOPSIS
 * The following functions summarize often needed procedures
 * in the mean square error kernel.
 * They can be found in the file named
 * kernel/mse_c/supportmse.c
 ***/

/****f* supportmse.c/support::create2Darray
 * SYNOPSIS
 * double **create2Darray(int ix, int iy)
 * DESCRIPTION
 * This function creates a 2 dimensional array with x lines
 * and y columns.
 ***/
double **create2Darray(int ix, int iy) {
   double **ppdarray = NULL;
   int ii = 0;

   IDL(1, printf("reached function create2Darray\n"));

   ppdarray = calloc(ix, sizeof(double *));
   for (ii = 0; ii < ix; ii++) {
      ppdarray[ii] = calloc(iy, sizeof(double));
   }

   IDL(1, printf("completed function create2Darray\n"));
   return ppdarray;
}

/****f* supportmse.c/support::free2Darray
 * SYNOPSIS
 * void free2Darray(double **ppdarray, int ix)
 * DESCRIPTION
 * function destructs a 2 dimensional array with x lines
 ***/
void free2Darray(double **ppdarray, int ix) {
   int ii = 0;

   IDL(1, printf("reached function free2Darray\n"));

   for (ii = 0; ii < ix; ii++) {
      free(ppdarray[ii]);
   }
   free(ppdarray);

   IDL(1, printf("completed function free2Darray\n"));
}

/****f* supportmse.c/support::outputmatrix
 * SYNOPSIS
 * double **create2Darray(int ix, int iy)
 * DESCRIPTION
 * This function prints a 2 dimensional array with x lines and
 * y columns to the screen (just needed for debugging purposes).
 ***/
void outputmatrix(double **ppdmatrix, int ix, int iy) {
   int ii = 0, ij = 0;

   IDL(1, printf("reached function outputmatrix\n"));

   for (ii = 0; ii < ix; ii++) {
      for (ij = 0; ij < iy; ij++) {
         printf(" %lf ", ppdmatrix[ii][ij]);
      }
      printf(" ii=%d \n", ii);
   }

   IDL(1, printf("completed function outputmatrix\n"));
}

