/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2011-01-27 fschmitt $
 * $URL: svn+ssh://benchit@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/BLAS2/C/CUDA/CUBLAS/sgemv/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Measurement of Nvidia CUBLAS sgemv performance
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

#define KERNELDES "sgemv"
#define LEGENDY "FLOPS (sgemv)"

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
    myinttype m, maxm;
    DT * hostData[3];
    DT * devData[3];
} mydata_t;

#endif

int setup(mydata_t *params, DT valueX, DT valueA)
{
    size_t i;
    CHECK_NULL(params);

    CUBLAS_CHECK(cublasInit());

    params->hostData[0] = (DT*) malloc(params->maxm * sizeof (DT));
    CHECK_NULL(params->hostData[0]);

    params->hostData[1] = (DT*) malloc(params->maxm * sizeof (DT));
    CHECK_NULL(params->hostData[1]);

    params->hostData[2] = (DT*) malloc(params->maxm * params->maxm * sizeof (DT));
    CHECK_NULL(params->hostData[2]);

    for (i = 0; i < params->maxm; i++) {
        params->hostData[0][i] = valueX;
    }

    for (i = 0; i < params->maxm * params->maxm; i++)
        params->hostData[2][i] = valueA;

    CUBLAS_CHECK(cublasAlloc(params->maxm, sizeof (DT), (void**) &(params->devData[0])));
    CUBLAS_CHECK(cublasAlloc(params->maxm, sizeof (DT), (void**) &(params->devData[1])));
    CUBLAS_CHECK(cublasAlloc(params->maxm * params->maxm, sizeof (DT), (void**) &(params->devData[2])));

    CUBLAS_CHECK(cublasSetVector(params->maxm, sizeof (DT), params->hostData[0], 1,
            (params->devData[0]), 1));

    CUBLAS_CHECK(cublasSetMatrix(params->maxm, params->maxm, sizeof (DT), params->hostData[2],
            params->maxm, params->devData[2], params->maxm));

    return 0;
}

int teardown(mydata_t *params)
{
    CUBLAS_CHECK(cublasFree(params->devData[0]));
    CUBLAS_CHECK(cublasFree(params->devData[1]));
    CUBLAS_CHECK(cublasFree(params->devData[2]));

    free(params->hostData[0]);
    free(params->hostData[1]);
    free(params->hostData[2]);

    CUBLAS_CHECK(cublasShutdown());
    return 0;
}

int run(mydata_t *params, DT alpha, DT beta)
{
    CUBLAS_CHECK(cublasSetVector(params->m, sizeof (DT), params->hostData[1], 1,
            (params->devData[1]), 1));

#ifdef PREC_SINGLE
    cublasSgemv('N', params->m, params->m, alpha, params->devData[2], params->m,
            params->devData[0], 1, beta, params->devData[1], 1);
#else
    cublasDgemv('N', params->m, params->m, alpha, params->devData[2], params->m,
            params->devData[0], 1, beta, params->devData[1], 1);
#endif

    CUBLAS_CHECK(cublasGetError());

    CUBLAS_CHECK(cublasGetVector(params->m, sizeof (DT), params->devData[1], 1,
            params->hostData[1], 1));

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
    infostruct->codesequence = bi_strdup("start kernel; matrix-vector multiplication; ");
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
    pmydata->maxm = m;

    IDL(3, printf("\nAllocate (%d,%d)-Matrix\n", pmydata->m, pmydata->m));

    if (setup(pmydata, 2.0, 0.5) != 0) {
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
        array[i] = (DT) 0.0;
    }
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
    double dstart = 0.0, dend = 0.0, dtime = 0.0;
    /* flops stores the calculated FLOPS */
    double dres = 0.0, dm = 0.0;
    /* ii is used for loop iterations */
    myinttype ii, imyproblemSize, numloop;
    /* cast void* pointer */
    mydata_t * pmydata = (mydata_t *) mdpv;

    DT alpha = 1.0;
    myinttype m, incxy = 1;

    /* get current problem size from problemlist */
    imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);
    m = imyproblemSize;

    pmydata->m = m;

    /* check wether the pointer to store the results in is valid or not */
    if (dresults == NULL) return 1;

    /* get the actual time
     * do the measurement / your algorithm
     * get the actual time
     */
    numloop = 1;
    do {
        initzero(pmydata->hostData[1], pmydata->m);

        dstart = bi_gettime();
        for (ii = 0; ii < numloop; ii++) {
            run(pmydata, alpha, alpha);
        }
        dend = bi_gettime();

        numloop <<= 3; /* numloop = numloop * 8 */
    } while (dend - dstart < 0.01);
    numloop >>= 3; /* undo last shift */

    if (abs((pmydata->hostData[1])[0]-(DT) m * (DT) numloop) > abs((DT) m)*(DT) numloop * EPSILON || abs((pmydata->hostData[1])[m - 1]-(DT) m * (DT) numloop) > abs((DT) m)*(DT) numloop * EPSILON) {
        IDL(0, printf("ERROR: the result is not valid!\n"));
        IDL(0, printf("expected: y(1)=%e, y(m)=%e,   got: y(1)=%e, y(m)=%e\n", (DT) m * (DT) numloop, (DT) m * (DT) numloop, (pmydata->hostData[1])[0], (pmydata->hostData[1])[m - 1]));
    }

    /* calculate the used time and FLOPS */
    dtime = dend - dstart;
    dtime -= dTimerOverhead;
    dm = (double) m;
    dres = dm * dm + (dm - 1) * dm; /* m*m mult and m-1*m add for A*x */
    dres += 2.0 * dm; /* m mult for alpha and m add for y */
    dres *= (DT) numloop;

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
