/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/float/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/
 
#include "sparse.h"

#include "sparseFormatJDS.h"
#include "sparseFormatCRS.h"
#include "sparseFormatCCS.h"

#include "matrix.h"
#include "vector.h"

#include "mpicomm.h"

/* Reads the environment variables used by this kernel. */
void evaluate_environment(mydata_t * pmydata)
{
   int errors = 0;
   char * p = 0;

   p = bi_getenv("BENCHIT_KERNEL_PERCENT", 0);
   if (p == NULL) errors++;
   else pmydata->percent = atof(p);
   p = bi_getenv("BENCHIT_KERNEL_SEED", 0);
   if (p == NULL) errors++;
   else pmydata->seed = atoi(p);

   p = bi_getenv("BENCHIT_KERNEL_INIT", 0);
   if (p == NULL) errors++;
   else pmydata->init = atoi(p);

   p = bi_getenv("BENCHIT_KERNEL_OUTPUT", 0);
   if (p == NULL) errors++;
   else pmydata->output = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_WITH_MATxVEC", 0);
   if (p == NULL) errors++;
   else pmydata->wMatxVec = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_VERIFY", 0);
   if (p == NULL) errors++;
   else {
     if (pmydata->wMatxVec) pmydata->verify = atoi(p);
     else pmydata->verify = 0;
   }

   if (errors > 0)
   {
      fprintf(stderr, "There's at least one environment variable not set!\n");
      exit(1);
   }
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   char *p = 0;
   mydata_t * penv;
   int i;
   
   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence = bi_strdup("generate matrix; converte matrix to JDS; multiply JDSxVector; converte matrix to CRS; multiply CRSxVector; converte matrix to CCS; multiply CCSxVector");
   infostruct->kerneldescription = bi_strdup("compare of storage formates for sparse matrices");
   infostruct->xaxistext = bi_strdup("Problem Size");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 1;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   if (penv->wMatxVec) { 
     infostruct->numfunctions = 4;
   } else {
     infostruct->numfunctions = 3;
   }

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   if (penv->output) {
      i = 0;
      if (penv->wMatxVec) {
        infostruct->yaxistexts[i] = bi_strdup("s");
        infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
        infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
        infostruct->legendtexts[i] = bi_strdup("time in s (std MATxVEC)");
        i++;
      }

      infostruct->yaxistexts[i] = bi_strdup("s");
      infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
      infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
      infostruct->legendtexts[i] = bi_strdup("time in s (JDSxVEC)");
      i++;

      infostruct->yaxistexts[i] = bi_strdup("s");
      infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
      infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
      infostruct->legendtexts[i] = bi_strdup("time in s (CRSxVEC)");
      i++;

      infostruct->yaxistexts[i] = bi_strdup("s");
      infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
      infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
      infostruct->legendtexts[i] = bi_strdup("time in s (CCSxVEC)");
   } else {
      i = 0;
      if (penv->wMatxVec) {
        infostruct->yaxistexts[i] = bi_strdup("FLOPS");
        infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
        infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
        infostruct->legendtexts[i] = bi_strdup("FLOPS (std MATxVEC)");
        i++;
      }

      infostruct->yaxistexts[i] = bi_strdup("FLOPS");
      infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
      infostruct->legendtexts[i] = bi_strdup("FLOPS (JDSxVEC)");
      i++;

      infostruct->yaxistexts[i] = bi_strdup("FLOPS");
      infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
      infostruct->legendtexts[i] = bi_strdup("FLOPS (CRSxVEC)");
      i++;

      infostruct->yaxistexts[i] = bi_strdup("FLOPS");
      infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
      infostruct->legendtexts[i] = bi_strdup("FLOPS (CCSxVEC)");
   }
  
   /* free all used space */
   if (penv) free(penv);
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
   evaluate_environment(pmydata);

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
  int i, j, mysize;
  int m, n, c_values;
  int * node_sizeOfValues;
  double percent;

  DT ** matrix;
  DT * vector;
  DT * b, * b0, * b1, * b2, * b3;
  DT eps1, eps2, eps3;
  JDS * jdsSparse;
  CRS * crsSparse;
  CCS * ccsSparse;

  /* flops stores the calculated FLOPS */
  double dres0 = 0.0;
  double dres = 0.0;
  /* dstart, dend: the start and end time of the measurement */
  /* dtime: the time for a single measurement in seconds */
  double * dstart, * dend, * dtime;

  /* ii is used for loop iterations */
  myinttype ii = 0, imyproblemSize = 0;
  /* cast void* pointer */
  mydata_t * pmydata = (mydata_t *) mdpv;

  MPI_Comm_size(MPI_COMM_WORLD, &globalsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &globalrank);

  if (pmydata->wMatxVec) {
    mysize = 4;
  } else {
    mysize = 3;
  }
  dstart = (double*)calloc(mysize, sizeof(double));
  dend   = (double*)calloc(mysize, sizeof(double));
  dtime  = (double*)calloc(mysize, sizeof(double));

  /* calculate real problemSize */
  imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);

  /* initial seed for the random generator */
  srand((unsigned int)pmydata->seed);

  /* check wether the pointer to store the results in is valid or not */
  if(globalrank==0) { if (dresults == NULL) return 1; }

  /* get the actual time
   * do the measurement / your algorythm
   * get the actual time
   */
  m = imyproblemSize;
  n = imyproblemSize;
  percent = pmydata->percent;

  /* create a new communicator for mpi (sometimes different from the MPI_COMM_WORLD, depend on the matrix size */
  createMyMPIComm(m);

  if(rank != MPI_UNDEFINED) {
    /* create / initialize matrix and vector */
    matrix = MPI_createMatrix(m, n);
    switch (pmydata->init) {
      case 1:  MPI_initIDENTITY(matrix, m, n);
               break;
      case 2:  MPI_initDIAG(matrix, m, n, 1);
               break;
      case 3:  MPI_init5PSTAR(matrix, m, n);
               break;
      default: MPI_initRandomMatrix(matrix, m, n, percent);
               break;
    }

    vector = MPI_createVector(n);
    if(rank==0) {
      MPI_initRandomVector(vector, n);
    }

    /* measure the time for Matrix*Vector, create a vector of the result to verify the results of the sparseformats */
    i = 0;
    if (pmydata->wMatxVec) {
      MPI_Barrier(mycomm);
      dstart[i] = bi_gettime();
        b = MPI_MatxVec(matrix, m, n, vector, n);
      MPI_Barrier(mycomm);
      dend[i] = bi_gettime();
      i++;
      if(rank==0) {
        if(pmydata->verify) {
          b0 = (DT*)malloc(m * sizeof(DT));
          MPI_copyVector(b, b0, m);
        }
        free(b);
      }
    }

    /* FLOPS of Matrix*Vector:
     * 1 addition and 1 multiplication per element of the matrix
     */
    dres0 = 2 * m * n;

    /* convert Matrix to JDS Format */
    jdsSparse = MPI_convertToJdsSparse(matrix, m, n);
  
    /* get the number of not-0-elements of the matrix,
     * calculate with that number the FLOPS,
     * for JDSxVec, CRSxVec and CCSxVec the FLOPS are the same:
     * 1 addition and 1 multiplication per not-0-element of the matrix
     */
    node_sizeOfValues = (int*)malloc(size * sizeof(int));
    MPI_Gather(&(jdsSparse->sizeOfValues), 1, MPI_iDT, node_sizeOfValues, 1, MPI_iDT, 0, mycomm);
    c_values = 0;
    if(rank==0) {
      for(j=0; j<size; j++) {
        c_values += node_sizeOfValues[j];
      }
    }
    dres = 2 * (double)c_values;
    free(node_sizeOfValues);

    /* measure the time for JDS*Vector, create a vector of the result to verify the results of the sparseformats */
    MPI_Barrier(mycomm);
    dstart[i] = bi_gettime();
      b = MPI_JDSxVec(jdsSparse, m, vector, n);
    MPI_Barrier(mycomm);
    dend[i] = bi_gettime();
    if(rank==0) {
      if(pmydata->verify) {
        b1 = (DT*)malloc(m * sizeof(DT));
        MPI_copyVector(b, b1, m);
      }
      free(b);
    }
    MPI_clearJdsSparse(jdsSparse);
    i++;

    /* convert Matrix to JDS Format */
    crsSparse = MPI_convertToCrsSparse(matrix, m, n);

    /* measure the time for CRS*Vector, create a vector of the result to verify the results of the sparseformats */
    MPI_Barrier(mycomm);
    dstart[i] = bi_gettime();
      b = MPI_CRSxVec(crsSparse, m, vector, n);
    MPI_Barrier(mycomm);
    dend[i] = bi_gettime();
    if(rank==0) {
      if(pmydata->verify) {
        b2 = (DT*)malloc(m * sizeof(DT));
        MPI_copyVector(b, b2, m);
      }
      free(b);
    }
    MPI_clearCrsSparse(crsSparse);
    i++;

    /* convert Matrix to JDS Format */
    ccsSparse = MPI_convertToCcsSparse(matrix, m, n);

    /* measure the time for CCS*Vector, create a vector of the result to verify the results of the sparseformats */
    MPI_Barrier(mycomm);
    dstart[i] = bi_gettime();
      b = MPI_CCSxVec(ccsSparse, m, vector, n);
    MPI_Barrier(mycomm);
    dend[i] = bi_gettime();
    if(rank==0) {
      if(pmydata->verify) {
        b3 = (DT*)malloc(m * sizeof(DT));
        MPI_copyVector(b, b3, m);
      }
      free(b);
    }
    MPI_clearCcsSparse(ccsSparse);

    /* clear the matrix/vector */
    MPI_clearMatrix(matrix);
    MPI_clearVector(vector);

    /* verify the correctness of the multiplication */  
    if(rank==0) {
      if(pmydata->verify) {
        eps1 = MPI_compareVector(b0, b1, m);
        eps2 = MPI_compareVector(b0, b2, m);
        eps3 = MPI_compareVector(b0, b3, m);
        if(eps1 != 0) printf("\nERROR: Result of MAT*VEC (m=%i, n=%i) is not the same like JDSxVEC, biggest distance=%e !",m,n,eps1);
        if(eps2 != 0) printf("\nERROR: Result of MAT*VEC (m=%i, n=%i) is not the same like CRSxVEC, biggest distance=%e !",m,n,eps2);
        if(eps3 != 0) printf("\nERROR: Result of MAT*VEC (m=%i, n=%i) is not the same like CCSxVEC, biggest distance=%e !",m,n,eps3);
        free(b0); free(b1); free(b2); free(b3);
      }
    }

    /* calculate the used time and FLOPS */
    for(i=0; i<mysize; i++) {
      dtime[i] = dend[i] - dstart[i];
      dtime[i] -= dTimerOverhead;
    }
      
    /* If the operation was too fast to be measured by the timer function,
     * mark the result as invalid 
     */
    for(i=0; i<mysize; i++) {
      if(dtime[i] < dTimerGranularity) dtime[i] = INVALID_MEASUREMENT;
    }

    /* store the results in results[1], results[2], ...
     * [1] for the first function, [2] for the second function
     * and so on ...
     * the index 0 always keeps the value for the x axis
     */
    if(rank==0) {
      if (pmydata->output) {
        i = 0;
        dresults[i] = (double)imyproblemSize; i++;
        if (pmydata->wMatxVec) {
          dresults[i] = dtime[i-1];
          i++;
        }
        dresults[i] = dtime[i-1]; i++;
        dresults[i] = dtime[i-1]; i++;
        dresults[i] = dtime[i-1];
      } else {
        i = 0;
        dresults[i] = (double)imyproblemSize; i++;
        if (pmydata->wMatxVec) {
          dresults[i] = dres0 / dtime[i-1];
          i++;
        }
        dresults[i] = dres / dtime[i-1]; i++;
        dresults[i] = dres / dtime[i-1]; i++;
        dresults[i] = dres / dtime[i-1];
      }
    }
  }

  /* clear the new communicator */
  freeMyMPIComm();

  free(dstart); free(dend); free(dtime);

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


