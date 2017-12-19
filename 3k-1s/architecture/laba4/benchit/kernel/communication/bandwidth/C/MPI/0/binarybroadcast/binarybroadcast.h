/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: binarybroadcast.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/bandwidth/C/MPI/0/binarybroadcast/binarybroadcast.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: bandwidth for a mpi broadcast implemented with a binary tree and send
 *******************************************************************/

#ifndef __binarybroadcast_h
#define __binarybroadcast_h

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

extern void mpibinarybroadcast_(void *, int *, int *, int *);

#endif


