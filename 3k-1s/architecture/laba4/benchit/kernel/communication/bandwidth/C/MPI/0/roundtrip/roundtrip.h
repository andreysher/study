/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: roundtrip.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/bandwidth/C/MPI/0/roundtrip/roundtrip.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Measure MPI bandwith for a round trip send algorithm
 *******************************************************************/


#ifndef __roundtrip_h
#define __roundtrip_h

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "interface.h"

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef int32_t myinttype;
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
   char *sendbuf;
   char *recvbuf;
   myinttype commrank, commsize;
   /* additional parameters */
   uint64_t maxsize;
} mydata_t;

#define MINTIME 1.0e-22

extern void entry_(void *mdpv, uint64_t * size);
extern void mpiroundtrip_(void *send, void *recv, uint64_t * size,
                          myinttype * rank, myinttype * commsize);

#endif

