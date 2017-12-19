/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: mse.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/meanSquareError/C/0/0/double/mse.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Solve a linear mean square error problem
 *******************************************************************/

#ifndef __mse_h
#define __mse_h

#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "interface.h"

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata {
   myinttype maxUnknowns;              /* number unknowns of the polynom */
   myinttype numPoints;                /* given points the polynom shall fit */
   double **ppdmatrix;
} mydata_t;

#define COEFF ((double)(rand()%1000))

#ifndef __mathfuncs_h
#define __mathfuncs_h
extern int signum(double dx);
extern double **matmul(double **ppdmatrix1, int ix1, int iy1,
                       double **ppdmatrix2, int ix2, int iy2, int ioffset);
extern double **createQ(double **ppdvector, int ilength, int ipos);
extern double *solve(double **ppdmatrix, int ivars);
#endif

#ifndef __supportmse_h
#define __supportmse_h
extern double **create2Darray(int ix, int iy);
extern void free2Darray(double **ppdarray, int ix);
extern void outputmatrix(double **ppdmatrix, int ix, int iy);
#endif

#endif

