/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: communicate.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/communicate/C/MPI/0/blocking/communicate.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare the needed time for different MPI communication methodes
 *******************************************************************/

#ifndef __communicate_h
#define __communicate_h

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
   myinttype commsize;
   myinttype commrank;
   /* additional parameters */
   myinttype maxsize;
} mydata_t;

extern void communicate(int, int, int, float *, int, double *);

#endif


