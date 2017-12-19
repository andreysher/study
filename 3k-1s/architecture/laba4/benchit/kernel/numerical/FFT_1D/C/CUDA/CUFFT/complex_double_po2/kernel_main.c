/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2011-01-29 fschmitt $
 * $URL: svn+ssh://benchit@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/FFT_1D/C/CUDA/CUFFT/complex_double_po2/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: 1D Fast Fourier Transform, Powers of 2,
 * double precision, complex data, CUFFT
 * (C language)
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <interface.h>
#include <fft.h>


int setup(mydata_t *params, DT_s valueR, DT_s valueI)
{
    size_t i;
    CHECK_NULL(params);

    size_t mem_size = sizeof (DT) * params->maxElements;

    params->hostData[0] = (DT*) malloc(mem_size);
    CHECK_NULL(params->hostData[0]);

    params->hostData[1] = (DT*) malloc(mem_size);
    CHECK_NULL(params->hostData[1]);

    for (i = 0; i < params->maxElements; i++) {
        params->hostData[0][i].x = valueR / (i + 1);
        params->hostData[0][i].y = valueI / (i + 1);
    }

    CUDA_CHECK(cudaMalloc((void**) &params->devData[0], mem_size));
    CUDA_CHECK(cudaMalloc((void**) &params->devData[1], mem_size));

    CUDA_CHECK(cudaMemcpy(params->devData[0], params->hostData[0], mem_size, cudaMemcpyHostToDevice));

    return 0;
}

int teardown(mydata_t *params)
{
    free(params->hostData[0]);
    free(params->hostData[1]);

    CUDA_CHECK(cudaFree(params->devData[0]));
    CUDA_CHECK(cudaFree(params->devData[1]));
    return 0;
}

int run(mydata_t *params)
{
    cufftHandle plan;
#ifdef PREC_SINGLE
    CUFFT_CHECK(cufftPlan1d(&plan, params->elements, CUFFT_C2C, 1));
#else
    CUFFT_CHECK(cufftPlan1d(&plan, params->elements, CUFFT_Z2Z, 1));
#endif

#ifdef PREC_SINGLE
    CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex *) params->devData[0], (cufftComplex *) params->devData[1], CUFFT_FORWARD));
#else
    CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex *) params->devData[0], (cufftDoubleComplex *) params->devData[1], CUFFT_FORWARD));
#endif

    CUFFT_CHECK(cudaMemcpy(params->hostData[1], params->devData[1], params->elements * sizeof (DT),
            cudaMemcpyDeviceToHost));

    CUFFT_CHECK(cufftDestroy(plan));

    return 0;
}

void bi_getinfo(bi_info* infostruct)
{
    char *p = 0;

    (void) memset(infostruct, 0, sizeof (bi_info));

    /* get environment variables for the kernel */
    /* parameter list */
    p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
    bi_parselist(p);

    /* get environment variables for the kernel */
    infostruct->codesequence = bi_strdup("cufft1dx");
    infostruct->kerneldescription = bi_strdup("1D Fast Fourier Transform, Powers of 2, double precision, complex data, CUFFT (C)");
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
    infostruct->yaxistexts[0] = bi_strdup("FLOPS");
    infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
    infostruct->base_yaxis[0] = 0;
    infostruct->legendtexts[0] = bi_strdup("FLOPS");

    /* this kernel needs a logarithmic x-axis */
    infostruct->base_xaxis = 2.0;
}

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

    pmydata->maxElements = (myinttype) pow(2, (myinttype) m);

    if (setup(pmydata, (DT_s) 1.1, (DT_s) 1.2) != 0) {
        fprintf(stderr, "Allocation of host/device data failed\n");
        fflush(stderr);
        exit(127);
    }

    return (void *) pmydata;
}

int bi_entry(void * mdpv, int iproblemSize, double * dresults)
{
    /* dstart, dend: the start and end time of the measurement */
    /* dtime: the time for a single measurement in seconds */
    double dstart = 0.0, dend = 0.0, dtime = 0.0, dinit = 0.0;
    /* flops stores the calculated FLOPS */
    double flops = 0.0;
    /* ii is used for loop iterations */
    myinttype ii, imyproblemSize, numberOfRuns;
    /* cast void* pointer */
    mydata_t* pmydata = (mydata_t*) mdpv;
    int invalid = 0;

    /* calculate real problemSize */
    imyproblemSize = (int) pow(2, (myinttype) bi_get_list_element(iproblemSize));
    pmydata->elements = imyproblemSize;

    /* store the value for the x axis in results[0] */
    dresults[0] = (double) imyproblemSize;

    /*** out of place run ***/
    numberOfRuns = 1;

    dstart = bi_gettime();
    /* fft calculation */
    run(pmydata);

    dend = bi_gettime();

    /* calculate the used time*/
    dtime = dend - dstart;
    dtime -= dTimerOverhead;

    /* loop calculation if accuracy is insufficient */
    while (dtime < 100 * dTimerGranularity) {

        numberOfRuns = numberOfRuns * 2;

        dstart = bi_gettime();
        for (ii = 0; ii < numberOfRuns; ii++) {
            /* fft calculation */
            run(pmydata);

        }
        dend = bi_gettime();

        /* calculate the used time*/
        dtime = dend - dstart;
        dtime -= dTimerOverhead;
    }

    /* if loop was necessary */
    if (numberOfRuns > 1) dtime = dtime / numberOfRuns;

    /* check for overflows */
    for (ii = 0; ii < imyproblemSize; ii++) {
        if (isnan(pmydata->hostData[1][ii].x) || isnan(pmydata->hostData[1][ii].y)) invalid = 1;
        if (isinf(pmydata->hostData[1][ii].x) || isinf(pmydata->hostData[1][ii].y)) invalid = 1;
    }

    /* calculate the used FLOPS */
    flops = (double) (5 * imyproblemSize * (log2(imyproblemSize)) / dtime);

    /* store the FLOPS in results[2] */
    if (invalid == 1) {
    	dresults[1] = INVALID_MEASUREMENT;
    	printf("Invalid measurement (problemSize = %d)\n", imyproblemSize);
    }
    else {
    	dresults[1] = flops;
    }

    return 0;
}

void bi_cleanup(void* mdpv)
{
    mydata_t* pmydata = (mydata_t*) mdpv;

    teardown(pmydata);

    if (pmydata) {
        free(pmydata);
    }
    return;
}
