/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/applications/CGV/C/MPI/0/double/kernel_main.c $
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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "interface.h"
#include "cgv.h"

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   char *p = 0;
   mydata_t * pmydata;
   
   pmydata = (mydata_t *) malloc(sizeof(mydata_t));

   MPI_Comm_rank(MPI_COMM_WORLD, &(pmydata->commrank));
   MPI_Comm_size(MPI_COMM_WORLD, &(pmydata->commsize));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

   IDL(3, printf("\nim bi_getinfo: rank=%d size=%d\n",pmydata->commrank, pmydata->commsize));

   infostruct->numfunctions = 1;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   infostruct->yaxistexts[0] = bi_strdup("FLOPS");
   infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[0] = 0;
   infostruct->legendtexts[0] = bi_strdup("FLOPS");

   infostruct->base_xaxis = 0.0;
   
   infostruct->codesequence = bi_strdup("");
   infostruct->kerneldescription = bi_strdup("CGV kernel");
   infostruct->xaxistext = bi_strdup("Problem Size");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = pmydata->commsize;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 1;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;

   /* free all used space */
   if (pmydata) free(pmydata);
}



/** Implementation of the bi_init of the BenchIT interface.
 *  Here you have the chance to allocate the memory you need.
 *  It is also possible to allocate the memory at the beginning
 *  of every single measurement and to free the memory thereafter.
 *  But always making use of the same memory is faster.
 *  HAVE A LOOK INTO THE HOWTO !
 */
void* bi_init(int problemSizemax)
{
   mydata_t * pmydata;

   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }
   
   MPI_Comm_rank(MPI_COMM_WORLD, &(pmydata->commrank));
   MPI_Comm_size(MPI_COMM_WORLD, &(pmydata->commsize));

   IDL(3, printf("\nim bi_init: rank=%d size=%d\n",pmydata->commrank, pmydata->commsize));
   return (void *)pmydata;
}



/** The central function within each kernel. This function
 *  is called for each measurement step seperately.
 *  @param  mdpv         a pointer to the structure created in bi_init,
 *                       it is the pointer the bi_init returns
 *  @param  problemSize  the actual problemSize
 *  @param  results      a pointer to a field of doubles, the
 *                       size of the field depends on the number
 *                       of functions, there are #functions+1
 *                       doubles
 *  @return 0 if the measurement was sucessfull, something
 *          else in the case of an error
 */
int bi_entry(void * mdpv, int iproblemSize, double * dresults)
{
  /* dstart, dend: the start and end time of the measurement */
  /* dtime: the time for a single measurement in seconds */
  double dstart = 0.0, dend = 0.0, dtime = 0.0, bi_flops;
  /* flops stores the calculated FLOPS */
  double dres = 0.0;
  /* ii is used for loop iterations */
  myinttype ii = 0, imyproblemSize = 0;
  /* cast void* pointer */
  mydata_t * pmydata = (mydata_t *) mdpv;

  IDL(3, printf("\nrank=%d entered bi_entry\n",pmydata->commrank));
  /* calculate real problemSize */
  imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);

  numprocs = pmydata->commsize;
  my_rank = pmydata->commrank;

/*
  Arguments:
  <number of rows in physical area of the Laplace equation>
  <number of columns in physical area of the Laplace equation>
  <maximal number of iterations>
  <epsilon>
  <print and debug level 0..5>
*/

  IDL(3, printf("\nim bi_entry, vor CGV-start: rank=%d size=%d\n",pmydata->commrank, pmydata->commsize));

  bi_flops = cgv_all((int)imyproblemSize, (int)imyproblemSize, 500, 1e-6, 0);

  IDL(3, printf("rank=%d Problemsize=%d, Value=%f\n",pmydata->commrank, imyproblemSize, dres));

  if (pmydata->commrank == 0)
  {
    /* store the value for the x axis in dresults[0] */
    dresults[0] = (double)imyproblemSize;
    dresults[1] = bi_flops;
  }

  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   mydata_t * pmydata = (mydata_t*)mdpv;
   if (pmydata) free(pmydata);
   return;
}

