/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: pingpong.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/latencies/C/MPI/0/pingpong/pingpong.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: pairwise Send/Recv between two MPI-Prozesses>
 *******************************************************************/

#ifndef __pingpong_h
#define __pingpong_h

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

#include "interface.h"
#include "mpi.h"

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata
{
   myinttype commsize;
   myinttype commrank;
   myinttype msgsize;
   myinttype repeat;
   myinttype pairs;
   myinttype * buffer;
} mydata_t;

extern void pingpong( int *from, int *to, void *mdpv );

#endif


