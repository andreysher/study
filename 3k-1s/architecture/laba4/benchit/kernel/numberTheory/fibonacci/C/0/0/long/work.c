/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numberTheory/fibonacci/C/0/0/long/work.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for calc of Fibonacci number (iterative / recursive)
 *******************************************************************/

#include "work.h"
#include "interface.h"

/****f* fib.c/measurement::recfib
 * SYNOPSIS
 * long recfib(long lnumber)
 * DESCRIPTION
 * This function calculates the fibonacci number of "lnumber" recursively.
 * Per definition is fib(n+1)=fib(n)+fib(n-1).
 ***/
long recfib(long lnumber) {
   IDL(1, printf("reached function recfib\n"));
   
   if (lnumber < 2) {
      return 1;
   }
   return (recfib(lnumber-1) + recfib(lnumber-2));
}

/****f* fib.c/measurement::linfib
 * SYNOPSIS
 * long linfib(long lnumber)
 * DESCRIPTION
 * This function calculates the fibonacci number of "lnumber" non-recursively
 * by using three variables.
 * Per definition is fib(n+1)=fib(n)+fib(n-1).
 ***/
long linfib(long lnumber) {
   long li, ln1, ln2;
   
   li  = 0;
   ln1 = 0;
   ln2 = 0;
   
   IDL(1, printf("reached function linfib\n"));

   if (lnumber < 2) {
      return 1;
   }
   
   ln1 = 1;
   ln2 = 1;
   
   for (li=0; li<lnumber-1; li++) {
      if (ln1 > ln2)
         ln2 = ln1 + ln2;
      else
         ln1 = ln2 + ln1;
   }
   
   IDL(1, printf("completed function linfib\n"));

   return ln1 > ln2 ? ln1 : ln2;
}

