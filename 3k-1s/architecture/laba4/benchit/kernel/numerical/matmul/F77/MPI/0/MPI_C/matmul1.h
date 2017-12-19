/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matmul1.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/F77/MPI/0/MPI_C/matmul1.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix multiply (Fortran 77) with MPI
 *******************************************************************/

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

#include <mpi.h>

#include "interface.h"
#include "fmc_h_machdefs.h"

#if ( defined ( _CRAY )    || \
      defined ( _SR8000 )  || \
      defined ( _USE_OLD_STYLE_CRAY_TYPE ) )
#define tstcas_ TSTCAS 
#endif

/* matmul1.h/definitions
 * SYNOPSIS
 * The variables that are described in the readme and the debuglevel.
 * SEE ALSO
 * kernel/matmul1_double
 ***/
#define MINTIME 1.0e-22

/*defining debuglevel*/
#ifndef DEBUGLEVEL
#define DEBUGLEVEL 0
#endif

/****is* matmul1.h/interface::float_data_struct
 * SYNOPSIS
 * typedef struct float_data_struct
 *    {
 *    double *pda, *pdb, *pdc;
 *    } fds
 * DESCRIPTION
 * structure saves both vectors that are added
 * result is saved in pdb
 ***/

typedef struct float_data_struct
   {
   double *pda, *pdb, *pdc;
   int MATMUL_START, MATMUL_INCREMENT, MATMUL_STEPS, MATMUL_PRECISION;
   } fds;
   
