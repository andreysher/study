/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: cannon.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/cannon/C/MPI/0/float/cannon.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: a MPI version of matrix-matrix multiplication
 *         (cannon algotithm)
 *******************************************************************/

#ifndef CANNON_H
#define CANNON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#include "interface.h"

#include "mpi.h"
int rank, globalrank;
int size, globalsize;
int tag;
MPI_Group MPI_GROUP_WORLD, mygroup;
MPI_Comm mycomm;

typedef struct {
   int row;
   int column;
} GridPosition;

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
   myinttype output;
} mydata_t;

#endif                                 // CANNON_H

