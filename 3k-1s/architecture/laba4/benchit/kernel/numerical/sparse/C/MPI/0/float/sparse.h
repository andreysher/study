/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparse.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/float/sparse.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#ifndef SPARSE_H
#define SPARSE_H


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include "interface.h"


#include "mpi.h"
int rank, globalrank;
int size, globalsize;
int tag;
MPI_Group MPI_GROUP_WORLD, mygroup;
MPI_Comm mycomm;

// #define DT int
// #define DT long
#define DT float
// #define DT double

// #define FORMAT1 "%i, "
// #define FORMAT2 "%i\n"
// #define FORMAT1 "%li, "
// #define FORMAT2 "%li\n"
#define FORMAT1 "%3.3e, "
#define FORMAT2 "%3.3e\n"
// #define FORMAT1 "%3.3le, "
// #define FORMAT2 "%3.3le\n"

// #define MPI_DT MPI_INT
// #define MPI_DT MPI_LONG
#define MPI_DT MPI_FLOAT
// #define MPI_DT MPI_DOUBLE

#define MPI_iDT MPI_INT


#ifndef DEBUG 
#define DEBUG 1
#endif

#define INFO 3
#define WARNING 2
#define ERROR 1

/*
IDL(ERROR, printf( "Error\n"));
IDL(WARNING, printf( "Warning\n"));
IDL(INFO, printf( "Info\n"));
*/

//#define IDL(X,Y) if((DEBUG)>=(X)){(void)(Y);fflush(stdout);fflush(stderr);}
#ifndef __work_h
#define __work_h

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
typedef struct mydata
{
   double percent;
   int seed;
   int init;
   int output;
   int wMatxVec;
   int verify;
} mydata_t;

#endif

#endif //SPARSE_H


