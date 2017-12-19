/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: inssort.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/sort/inssort/C/0/0/simple/inssort.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Evaluate time to sort int/float array with insertion sort
 *******************************************************************/

#include "inssort.h"
#include "interface.h"

/****f* inssort_funcs.c/measurement
 * SYNOPSIS
 * The following functions are used for measuring and veryfying
 * the inssort algorithm.
 * They can be found in the file named
 * kernel/inssort_c/inssort_funcs.c
 ***/

/****f* inssort_funcs.c/measurement::inssorti
 * SYNOPSIS
 * void inssorti(int *pisort, long lnumber)
 * DESCRIPTION
 * This function makes inssort on an array of integers (pisort) with
 * a size of "lnumber".
 ***/
void inssorti(int *pisort, long lnumber) {
   /* variables for saving an element, position in the array and for
    * loops and buffering */
   int inwork=0, ipos=0, ii=0;
   
   /* search smallest element and bring it to the 1st position */
   inwork = pisort[0];
   ipos = 0;
   for (ii=0; ii<lnumber; ii++) {
      if (inwork > pisort[ii]) {
         inwork = pisort[ii];
         ipos = ii;
      }
   }
   /* inwork is smallest element and pos is position in sort
    * ->changing elements */
   inwork = pisort[0];
   pisort[0] = pisort[ipos];
   pisort[ipos] = inwork;
   
   /* begin of insertion sort */
   for (ii=1; ii<lnumber; ii++) {
      /* inwork is the actual number */
      inwork = pisort[ii];
      /* actual position */
      ipos = ii;
      /* as long as the element before is smaller
       * -> move the element to next position */
      /* if sort[0] wouldnt be smallest element this wouldnt work! */
      while (pisort[ipos-1]>inwork) {
         pisort[ipos] = pisort[ipos - 1];
         ipos = ipos - 1;
      }
      /* write the actual number to the correct position */
      pisort[ipos] = inwork;
   }
}

/****f* inssort_funcs.c/measurement::inssortf
 * SYNOPSIS
 * void inssortf(int *pfsort, long lnumber)
 * DESCRIPTION
 * This function makes inssort on an array of floats (pfsort) with
 * a size of "lnumber".
 ***/
void inssortf(float *pfsort, long lnumber) {
   /* variables for position in the array and for loops and buffering */
   int ipos=0, ii=0;
   /* variable for saving an element */
   float finwork=0.0;
   
   /* search smallest element and bring it to the 1st position */
   finwork = pfsort[0];
   ipos = 0;
   for (ii=0; ii<lnumber; ii++) {
      if (finwork > pfsort[ii]) {
         finwork = pfsort[ii];
         ipos = ii;
      }
   }
   /* inwork is smallest element and pos is position in sort
    * ->changing elements */
   finwork = pfsort[0];
   pfsort[0] = pfsort[ipos];
   pfsort[ipos] = finwork;
   
   /* begin of insertion sort */
   for (ii=1; ii<lnumber; ii++) {
      /* inwork is the actual number */
      finwork = pfsort[ii];
      /* actual position */
      ipos = ii;
      /* as long as the element before is smaller
       * -> move the element to next position */
      /* if sort[0] wouldnt be smallest element this wouldnt work! */
      while (pfsort[ipos-1]>finwork) {
         pfsort[ipos] = pfsort[ipos - 1];
         ipos = ipos - 1;
      }
      /* write the actual number to the correct position */
      pfsort[ipos] = finwork;
   }
}

/****f* inssort_funcs.c/measurement::verifyi
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

/****f* inssort_funcs.c/measurement::verifyf
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

