/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2010-01-25 fschmitt $
 * $URL: svn+ssh://benchit@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/BLAS1/C/CUDA/CUBLAS/saxpy/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Measurement of Nvidia CUBLAS saxpy performance
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"

#include "kernel_blas.h"

#define PREC_SINGLE

#ifdef PREC_SINGLE
#define DT float
#define EPSILON 1.0e-6
#else
#define DT double
#define EPSILON 1.0e-15
#endif

#define KERNELDES "saxpy"
#define LEGENDY "FLOPS (saxpy)"

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
typedef struct
{
    myinttype elements;
    myinttype maxElements;
    DT * hostData[2];
    DT * devData[2];
} mydata_t;

#endif

int setup(mydata_t *params, DT value)
{
    size_t i;
    CHECK_NULL(params);
    
    cudaSetDevice(1);

    CUBLAS_CHECK(cublasInit());
    
    

    params->hostData[0] = (DT*) malloc(params->maxElements * sizeof (DT));
    CHECK_NULL(params->hostData[0]);

    params->hostData[1] = (DT*) malloc(params->maxElements * sizeof (DT));
    CHECK_NULL(params->hostData[1]);

    for (i = 0; i < params->maxElements; i++) {
        params->hostData[0][i] = value;
    }

    CUBLAS_CHECK(cublasAlloc(params->maxElements, sizeof (DT), (void**) &(params->devData[0])));
    CUBLAS_CHECK(cublasAlloc(params->maxElements, sizeof (DT), (void**) &(params->devData[1])));

    CUBLAS_CHECK(cublasSetVector(params->maxElements, sizeof (DT), params->hostData[0], 1,
            (params->devData[0]), 1));

    return 0;
}

int teardown(mydata_t *params)
{
    CUBLAS_CHECK(cublasFree(params->devData[0]));
    CUBLAS_CHECK(cublasFree(params->devData[1]));

    free(params->hostData[0]);
    free(params->hostData[1]);

    CUBLAS_CHECK(cublasShutdown());
    return 0;
}

int run(mydata_t *params, DT alpha)
{
    CUBLAS_CHECK(cublasSetVector(params->elements, sizeof (DT),
            params->hostData[1], 1, params->devData[1], 1));

#ifdef PREC_DOUBLE
    cublasDaxpy(params->elements, alpha,
            params->devData[0], 1, params->devData[1], 1);
#else
    cublasSaxpy(params->elements, alpha,
            params->devData[0], 1, params->devData[1], 1);
#endif

    CUBLAS_CHECK(cublasGetError());

    CUBLAS_CHECK(cublasGetVector(params->elements, sizeof (DT),
            params->devData[1], 1, params->hostData[1], 1));

    return 0;
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
    char * p = 0;
    mydata_t * penv;

    (void) memset(infostruct, 0, sizeof (bi_info));
    penv = (mydata_t *) malloc(sizeof (mydata_t));

    /* get environment variables for the kernel */
    /* parameter list */
    p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
    bi_parselist(p);

    /* get environment variables for the kernel */
    infostruct->codesequence = bi_strdup("start kernel; y = alpha * x + y with vectors x and y; ");
    infostruct->kerneldescription = bi_strdup(KERNELDES);
    infostruct->xaxistext = bi_strdup("Problem Size");
    infostruct->num_measurements = infostruct->listsize;
    infostruct->num_processes = 1;
    infostruct->num_threads_per_process = 0;
    infostruct->kernel_execs_mpi1 = 0;
    infostruct->kernel_execs_mpi2 = 0;
    infostruct->kernel_execs_pvm = 0;
    infostruct->kernel_execs_omp = 0;
    infostruct->kernel_execs_pthreads = 0;
    infostruct->numfunctions = 1;

    /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
    infostruct->yaxistexts[0] = bi_strdup("flop/s");
    infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
    infostruct->base_yaxis[0] = 0; //logarythmic axis 10^x
    infostruct->legendtexts[0] = bi_strdup(LEGENDY);

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
    myinttype i, m, tmp;

    pmydata = (mydata_t*) malloc(sizeof (mydata_t));
    if (pmydata == 0) {
        fprintf(stderr, "Allocation of structure mydata_t failed\n");
        fflush(stderr);
        exit(127);
    }

    m = (myinttype) bi_get_list_element(1);
    for (i = 2; i <= problemSizemax; i++) {
        tmp = (myinttype) bi_get_list_element(i);
        if (tmp > m) m = tmp;
    }
    pmydata->maxElements = m;

    IDL(3, printf("\nAllocate vector with size %d\n", m));

    /* fill x with 0.5 (alpha will be 2.0 -> y(i) = 1.0)
     */
    if (setup(pmydata, 0.5) != 0) {
        fprintf(stderr, "Allocation of host/device data failed\n");
        fflush(stderr);
        exit(127);
    }

    return (void *) pmydata;
}

/** initialize array with 0
 */
void initzero(DT *array, int size)
{
    myinttype i;

    for (i = 0; i < size; i++) {
        array[i] = 0.0;
    }
}

/** The central function within each kernel. This function
 *  is called for each measurement step seperately.
 *  @param  mdpv         a pointer to the structure created in bi_init,
 *                       it is the pointer the bi_init returns
 *  @param  problemSize  the actual problemSize
 *  @param  results      a pointer to a field of DTs, the
 *                       size of the field depends on the number
 *                       of functions, there are #functions+1
 *                       DTs
 *  @return 0 if the measurement was sucessfull, something
 *          else in the case of an error
 */
int bi_entry(void * mdpv, int iproblemSize, double * dresults)
{
    /* dstart, dend: the start and end time of the measurement */
    /* dtime: the time for a single measurement in seconds */
    double dstart = 0.0, dend = 0.0, dtime = 0.0;
    /* flops stores the calculated FLOPS */
    double dres = 0.0, dm = 0.0;
    /* ii is used for loop iterations */
    myinttype ii, imyproblemSize, numloop;
    /* cast void* pointer */
    mydata_t * pmydata = (mydata_t *) mdpv;

    DT alpha = 2.0;
    myinttype m, incxy = 1;

    /* get current problem size from problemlist */
    imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);
    m = imyproblemSize;

    pmydata->elements = m;

    /* check wether the pointer to store the results in is valid or not */
    if (dresults == NULL) return 1;

    /* get the actual time
     * do the measurement / your algorithm
     * get the actual time
     */
    numloop = 1;
    do {
        initzero(pmydata->hostData[1], pmydata->elements);

        dstart = bi_gettime();
        for (ii = 0; ii < numloop; ii++)
            run(pmydata, alpha);
        dend = bi_gettime();

        numloop <<= 3; /* numloop = numloop * 8 */
    } while (dend - dstart < 0.01);
    numloop >>= 3; /* undo last shift */

    if (abs((pmydata->hostData[1])[0]-(DT) numloop) > (DT) numloop * EPSILON || abs((pmydata->hostData[1])[m - 1]-(DT) numloop) > (DT) numloop * EPSILON) {
        IDL(0, printf("ERROR: the result is not valid!\n"));
        IDL(0, printf("expected: y(1)=%e, y(m)=%e,   got: y(1)=%e, y(m)=%e\n", (DT) numloop, (DT) numloop, (pmydata->hostData[1])[0], (pmydata->hostData[1])[m - 1]));
    }

    /* calculate the used time and FLOPS */
    dtime = dend - dstart;
    dtime -= dTimerOverhead;
    dm = (double) m;
    dres = dm; /* m mult for alpha*x */
    dres += dm; /* m add for y */
    dres *= (double) numloop;

    /* If the operation was too fast to be measured by the timer function,
     * mark the result as invalid
     */
    if (dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;

    /* store the results in results[1], results[2], ...
     * [1] for the first function, [2] for the second function
     * and so on ...
     * the index 0 always keeps the value for the x axis
     */
    dresults[0] = (double) imyproblemSize;
    dresults[1] = (dtime != INVALID_MEASUREMENT) ? dres / dtime : INVALID_MEASUREMENT;

    return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
    mydata_t * pmydata = (mydata_t*) mdpv;

    teardown(pmydata);

    if (pmydata) free(pmydata);
    return;
}
