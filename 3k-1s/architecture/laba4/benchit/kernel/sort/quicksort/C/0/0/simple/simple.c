/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: simple.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/sort/quicksort/C/0/0/simple/simple.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "simple.h"
#include "interface.h"


/**
 * compare-function needed by the clib-qsort
 * integer-elements
 */
int quicksort_clib_myinttype(const void *pvelement1, const void *pvelement2)
{
   myinttype  ii, ij;

   /*initialize variables*/
   ii=0;
   ij=0;

   /*debugging level 1: mark begin and end of function*/
   if (DEBUGLEVEL > 0)
      {
      printf("reached function quicksort_clib_myinttype\n");
      fflush(stdout);
      }

   /*void pomyinttypeers must be casted*/
   ii = (myinttype) (*(myinttype *) pvelement1);
   ij = (myinttype) (*(myinttype *) pvelement2);

   if (ii > ij)
      {
      /*debugging level 1: mark begin and end of function*/
      if (DEBUGLEVEL > 0)
         {
         printf("completed function quicksort_clib_myinttype\n");
         fflush(stdout);
         }
      return 1;
      }
   if (ii < ij)
      {
      /*debugging level 1: mark begin and end of function*/
      if (DEBUGLEVEL > 0)
         {
         printf("completed function quicksort_clib_myinttype\n");
         fflush(stdout);
         }
      return -1;
      }
   /*debugging level 1: mark begin and end of function*/
   if (DEBUGLEVEL > 0)
      {
      printf("completed function quicksort_clib_myinttype\n");
      fflush(stdout);
      }
   return 0;
}

/**
 * compare-function needed by the clib-qsort
 * float-elements
 */
int quicksort_clib_flt(const void *pvelement1, const void *pvelement2)
{
   float fi, fj;

   /*initialize variables*/
   fi=0;
   fj=0;

   /*debugging level 1: mark begin and end of function*/
   if (DEBUGLEVEL > 0)
      {
      printf("reached function quicksort_clib_flt\n");
      fflush(stdout);
      }

   /*void pointers must be casted*/
   fi = (float) (*(float *) pvelement1);
   fj = (float) (*(float *) pvelement2);

   if (fi > fj)
      {
      /*debugging level 1: mark begin and end of function*/
      if (DEBUGLEVEL > 0)
         {
         printf("completed function quicksort_clib_flt\n");
         fflush(stdout);
         }
      return 1;
      }
   if (fi < fj)
      {
         /*debugging level 1: mark begin and end of function*/
      if (DEBUGLEVEL > 0)
         {
         printf("completed function quicksort_clib_flt\n");
         fflush(stdout);
         }
         return -1;
      }
   /*debugging level 1: mark begin and end of function*/
   if (DEBUGLEVEL > 0)
      {
      printf("completed function quicksort_clib_flt\n");
      fflush(stdout);
      }
   return 0;
}


/**
 * compare-function needed by the clib-qsort
 * double-elements
 */
int quicksort_clib_dbl(const void *pvelement1, const void *pvelement2)
{
   double fi, fj;

   /*initialize variables*/
   fi=0;
   fj=0;

   /*debugging level 1: mark begin and end of function*/
   if (DEBUGLEVEL > 0)
      {
      printf("reached function quicksort_clib_dbl\n");
      fflush(stdout);
      }

   /*void pointers must be casted*/
   fi = (double) (*(double *) pvelement1);
   fj = (double) (*(double *) pvelement2);

   if (fi > fj)
      {
      /*debugging level 1: mark begin and end of function*/
      if (DEBUGLEVEL > 0)
         {
         printf("completed function quicksort_clib_dbl\n");
         fflush(stdout);
         }
      return 1;
      }
   if (fi < fj)
      {
         /*debugging level 1: mark begin and end of function*/
      if (DEBUGLEVEL > 0)
         {
         printf("completed function quicksort_clib_dbl\n");
         fflush(stdout);
         }
         return -1;
      }
   /*debugging level 1: mark begin and end of function*/
   if (DEBUGLEVEL > 0)
      {
      printf("completed function quicksort_clib_dbl\n");
      fflush(stdout);
      }
   return 0;
}



int verify_int(myinttype *pfprobe, long lelements)
   {
   int ii;

   /*initialize variables*/
   ii = 0;

/*any element on position n+1 has to be larger
  or equal to element on position n...*/
   for (ii = 1; ii < lelements; ii++)
      {
      if (pfprobe[ii - 1] > pfprobe[ii])
         {
         return 0;
         }
      }

   /*"1" means success */
   return 1;
   }



int verify_float(float *pfprobe, long lelements)
   {
   int ii;

   /*initialize variables*/
   ii = 0;

/*any element on position n+1 has to be larger
  or equal to element on position n...*/
   for (ii = 1; ii < lelements; ii++)
      {
      if (pfprobe[ii - 1] > pfprobe[ii])
         {
         return 0;
         }
      }

   /*"1" means success */
   return 1;
   }



int verify_double(double *pfprobe, long lelements)
   {
   int ii;

   /*initialize variables*/
   ii = 0;

/*any element on position n+1 has to be larger
  or equal to element on position n...*/
   for (ii = 1; ii < lelements; ii++)
      {
      if (pfprobe[ii - 1] > pfprobe[ii])
         {
         return 0;
         }
      }

   /*"1" means success */
   return 1;
   }



void quicksort_wikipedia_int(int * a, int al, int ar) {
        int links=al, rechts=ar, pivo=a[(al+ar)/2], tmp;
        do {
                while(a[links]<pivo) links++;
                while(a[rechts]>pivo) rechts--;
                if (links <= rechts) {
                        tmp=a[links];
                        a[links]=a[rechts];
                        a[rechts]=tmp;
                        links++;
                        rechts--;
                }       
        } while(links<rechts);
        if (al < rechts) quicksort_wikipedia_int(a, al, rechts);
        if (links < ar) quicksort_wikipedia_int(a, links, ar);
}



void quicksort_wikipedia_flt(float * a, int al, int ar) {
        int links=al, rechts=ar; 
		float pivo=a[(al+ar)/2], tmp;
        do {
                while(a[links]<pivo) links++;
                while(a[rechts]>pivo) rechts--;
                if (links <= rechts) {
                        tmp=a[links];
                        a[links]=a[rechts];
                        a[rechts]=tmp;
                        links++;
                        rechts--;
                }       
        } while(links<rechts);
        if (al < rechts) quicksort_wikipedia_flt(a, al, rechts);
        if (links < ar) quicksort_wikipedia_flt(a, links, ar);
}



void quicksort_wikipedia_dbl(double * a, int al, int ar) {
        int links=al, rechts=ar; 
		double pivo=a[(al+ar)/2], tmp;
        do {
                while(a[links]<pivo) links++;
                while(a[rechts]>pivo) rechts--;
                if (links <= rechts) {
                        tmp=a[links];
                        a[links]=a[rechts];
                        a[rechts]=tmp;
                        links++;
                        rechts--;
                }       
        } while(links<rechts);
        if (al < rechts) quicksort_wikipedia_dbl(a, al, rechts);
        if (links < ar) quicksort_wikipedia_dbl(a, links, ar);
}

