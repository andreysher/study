/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: selsort.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/sort/selsort/C/0/0/simple/selsort.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Evaluate time to sort int/float array with selection sort
 *******************************************************************/

#include "selsort.h"
#include "interface.h"

/****f* selsort_funcs.c/measurement
 * SYNOPSIS
 * The following functions are used for measuring and veryfying
 * the selsort algorithm.
 * They can be found in the file named
 * kernel/selsort_c/selsort_funcs.c
 ***/

/****f* selsort_funcs.c/measurement::selsorti
 * SYNOPSIS
 * void selsorti(int *pisort, long lnumber)
 * DESCRIPTION
 * This function makes selsort on an array of integers (pisort) with
 * a size of "lnumber".
 ***/
void selsorti(int *pisort, long lnumber) {
   /* variables for buffering and looping */
   int ih=0, ii=0, ij=0, imin=0;
   
   /* search the smallest element for all remaining in array */
   for (ii=0; ii<lnumber; ii++) {
      /* position of first element of the remaining array */
      imin = ii;
      /* searching the smallest element in the remaining array */
      for (ij=ii; ij<lnumber; ij++) {
         if (pisort[ij] < pisort[imin])
            imin = ij;
      }
      /* change the first element in the remaining array
       * with the smallest one */
      ih = pisort[imin];
      pisort[imin] = pisort[ii];
      pisort[ii] = ih;
   }
}

/****f* selsort_funcs.c/measurement::selsortf
 * SYNOPSIS
 * void selsortf(int *pfsort, long lnumber)
 * DESCRIPTION
 * This function makes selsort on an array of floats (pfsort) with
 * a size of "lnumber".
 ***/
void selsortf(float *pfsort, long lnumber) {
   /* variables for looping and buffering */
   int ii=0, ij=0, imin=0;
   /* variables for buffering */
   float fh=0.0;
   
   /* search the smallest element for all remaining in arrray */
   for (ii=0; ii<lnumber; ii++) {
      /* position of first element of the remaining array */
      imin = ii;
      /* searching the smallest element in the remaining array */
      for (ij=ii; ij<lnumber; ij++) {
         if (pfsort[ij] < pfsort[imin])
            imin = ij;
      }
      /* change the first element in the remaining array
       * with the smallest one */
      fh = pfsort[imin];
      pfsort[imin] = pfsort[ii];
      pfsort[ii] = fh;
   }
}

/****f* selsort_funcs.c/measurement::verifyi
 * SYNOPSIS
 * int verifyi(int *piprobe, long lelements)
 * DESCRIPTION
 * This function tests if an array of integers (piprobe) with
 * a size of "lelements" has been sorted correctly.
 ***/
int verifyi(int *piprobe, long lelements) {
   int ii=0;
   
   /* any element on position n+1 has to be larger
    * or equal to element on position n... */
   for (ii=1; ii<lelements; ii++) {
      /* if not -> "0" means failure */
      if (piprobe[ii - 1] > piprobe[ii])
         return 0;
   }
   
   /* "1" means success */
   return 1;
}

/****f* selsort_funcs.c/measurement::verifyf
 * SYNOPSIS
 * int verifyf(int *pfprobe, long lelements)
 * DESCRIPTION
 * This function tests if an array of floats (pfprobe) with
 * a size of "lelements" has been sorted correctly.
 ***/
int verifyf(float *pfprobe, long lelements) {
   int ii=0;
   
   /* any element on position n+1 has to be larger
    * or equal to element on position n... */
   for (ii=1; ii<lelements; ii++) {
      if (pfprobe[ii - 1] > pfprobe[ii])
         return 0;
   }
   
   /* "1" means success */
   return 1;
}

