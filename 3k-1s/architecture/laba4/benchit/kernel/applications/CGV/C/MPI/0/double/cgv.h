/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: cgv.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/applications/CGV/C/MPI/0/double/cgv.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Heat conduction (CG-Solver, vectorized)
 *         Based on the CGV-Exmple in the Parallel Programming Workshop
 *         http://www.hlrs.de/organization/par/par_prog_ws/
 *         Section [42] "Laplace-Example with MPI and PETSc", and
 *         http://www.hlrs.de/organization/par/par_prog_ws/practical/README.html
 *         CG-Solver, vectorized - CGV.tar.gz
 *         by Dr. Rolf Rabenseifner (HLRS)
 *******************************************************************/


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
   myinttype commsize;
   myinttype commrank;
} mydata_t;

extern double simple( myinttype * );

#endif


#ifndef serial
#  define TRUE    1
#  define FALSE   0
#  define MSG_TAG 333
#endif

#ifndef serial
#  include <mpi.h>
#else
#  include <time.h>
#endif

/* options: */
int     m,n;         /* size of the physical-heat matrix without borders */
                     /* m=vertical, n=horizontal */
int     iter_max;    /* maximal number of iterations in CG */
double  epsilon;     /* Limit for sqr( || residual vector || ) */
int     print_level; /* print_level = 1: Only result-error and partial result matrix */
                                 /* = 2: and residual norm after each iteration */
                                 /* = 3: and result physical-heat matrix */
                                 /* = 4: and all vector and halo data in first iteration */
                                 /* = 5: and all vector and halo data in _all_ iterations */
int     decomp_dims; /* dimensins of the domain decomposition */
int     m_procs, n_procs; /* number of processes in both directions of the 2-dim domain decomposition */
int     mprt, nprt;  /* number of rows and columns when printing the partial result matrix */

/* global structures and data: */

#ifndef serial
 MPI_Comm comm_cart;
#endif
int numprocs, my_rank;
unsigned long flops;



double cgv_all(int bi_m, int bi_n, int bi_iter_max, int bi_epsilon, int bi_print_level); 


