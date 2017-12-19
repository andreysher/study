/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/FFT_3D/C/OpenMP/MKL/complex_float_npo2/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: 3D Fast Fourier Transform, Non-Powers of 2,
 * single precision, complex data, MKL
 * (C language), OpenMP version
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <interface.h>
#include <fft.h>
#include <omp.h>
/*included for version purposes*/
#include <mkl_blas.h>

int functionCount;
int valuesPerFunction;


void initData_ip(mydata_t* mdpv,int problemSize)
{
   int i;
   for (i = 0; i < problemSize * problemSize * problemSize; i++)
   {
      mdpv->inout[2*i] = 0.0;
      mdpv->inout[2*i+1] = 0.0;
   }
}


void initData_oop(mydata_t* mdpv,int problemSize)
{
   int i;
   for (i = 0; i < problemSize * problemSize * problemSize; i++)
   {
      mdpv->in[2*i] = 1.1/(i+1);
      mdpv->in[2*i+1] = 1.2/(i+1);
      mdpv->out[2*i] = 0.0;
      mdpv->out[2*i+1] = 0.0;
   }
}


/* Reads the environment variables used by this kernel. */
void evaluate_environment(mydata_t* pmydata)
{
   char* p = 0;
   myinttype nMeasurements;
   int i;

   p = getenv("BENCHIT_KERNEL_PROBLEMSIZES");
   if (p != NULL)  {
      nMeasurements = 1;
      while (p) {
         p = strstr(p, ",");
         if (p) {
            p++;
            nMeasurements++;
         }
      }
      pmydata->problemSizes = (myinttype *)malloc(sizeof(myinttype) * nMeasurements);

      p = getenv("BENCHIT_KERNEL_PROBLEMSIZES");
      pmydata->problemSizes[0] = atof(p);

      for (i = 1; i < nMeasurements; i++) {
         p = strstr(p, ",") + 1;
         pmydata->problemSizes[i] = atof(p);
      }
   }
   else {
      pmydata->problemSizes = NULL;
   }

   pmydata->min = pmydata->problemSizes[0];
   pmydata->max = pmydata->problemSizes[nMeasurements-1];
   pmydata->steps = nMeasurements;
}


void bi_getinfo(bi_info* infostruct)
{
   mydata_t* penv;

   char buf[150];
   char version[100];
   int loop;

   penv = (mydata_t*)malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   evaluate_environment(penv);
   infostruct->codesequence = bi_strdup("DftiComputeForward()");
   infostruct->kerneldescription = bi_strdup("3D Fast Fourier Transform, Non-Powers of 2, single precision, complex data, MKL (C), OpenMP version");
   infostruct->xaxistext = bi_strdup("Problem Size");
   infostruct->num_measurements = penv->steps;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 1;
   infostruct->kernel_execs_pthreads = 0;

   functionCount = 2; /* in place, out of place */
   valuesPerFunction = 1; /* FLOPS (calculated) */
   infostruct->numfunctions = functionCount * valuesPerFunction;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   infostruct->yaxistexts[0] = bi_strdup("FLOPS");
   infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[0] = 0;
   infostruct->legendtexts[0] = bi_strdup("FLOPS in place");
   infostruct->yaxistexts[1] = bi_strdup("FLOPS");
   infostruct->selected_result[1] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[1] = 0;
   infostruct->legendtexts[1] = bi_strdup("FLOPS out of place");

   /* this kernel needs a logarithmic x-axis */
   infostruct->base_xaxis = 2.0;

   /*get MKL version*/
   MKLGetVersionString(version,100);
   for (loop=0;loop<100;loop++)
   {
      if (version[loop]=='\0') break;
   }
   if (loop<100) sprintf(buf,"mklversion=\"%s\"",version);
   else sprintf(buf,"mklversion=\"\"");
   infostruct->additional_information=bi_strdup(buf);

   /* free all used space */
   if (penv)free(penv);
}


void* bi_init(int problemSizemax)
{
   mydata_t* pmydata;

   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr); exit(127);
   }
   else {
      evaluate_environment(pmydata);
      problemSizemax = (int)pmydata->max;
   }
   return (void*)pmydata;
}


int bi_entry(void * mdpv, int iproblemSize, double * dresults)
{
   /* dstart, dend: the start and end time of the measurement */
   /* dtime: the time for a single measurement in seconds */
   double dstart = 0.0, dend = 0.0, dtime = 0.0, dinit = 0.0;
   /* flops stores the calculated FLOPS */
   double flops = 0.0;
   /* ii is used for loop iterations */
   myinttype ii, jj, imyproblemSize, numberOfRuns;
   /* cast void* pointer */
   mydata_t* pmydata = (mydata_t*)mdpv;
   int invalid = 0;
   long status, len[3];

   /* calculate real problemSize */
   imyproblemSize = (int)(pmydata->problemSizes[iproblemSize - 1]);

   len[0] = imyproblemSize;
   len[1] = imyproblemSize;
   len[2] = imyproblemSize;

   /* store the value for the x axis in results[0] */
   dresults[0] = (double)imyproblemSize;


   /*** in place run ***/

   /* malloc */
   pmydata->inout = (float*)malloc(sizeof(float) * imyproblemSize * imyproblemSize * imyproblemSize * 2);

   /* create FFT plan */
   status = DftiCreateDescriptor(&pmydata->my_desc_handle, DFTI_SINGLE, DFTI_COMPLEX, 3, len);
   status = DftiCommitDescriptor(pmydata->my_desc_handle);


   /* init stuff */
   initData_ip(pmydata, imyproblemSize);

   numberOfRuns = 1;

   dstart = bi_gettime();
   /* fft calculation */
   status = DftiComputeForward(pmydata->my_desc_handle, pmydata->inout);
   dend = bi_gettime();

   /* calculate the used time*/
   dtime = dend - dstart;
   dtime -= dTimerOverhead;

   /* loop calculation if accuracy is insufficient */
   while (dtime < 100 * dTimerGranularity) {

     numberOfRuns = numberOfRuns * 2;

     dstart = bi_gettime();
     for (jj = 0; jj < numberOfRuns; jj++) {
       /* fft calculation */
       status = DftiComputeForward(pmydata->my_desc_handle, pmydata->inout);
     }
     dend = bi_gettime();

     dtime = dend - dstart;
     dtime -= dTimerOverhead;
   }

   /* check for overflows */
   for (ii = 0; ii < imyproblemSize * imyproblemSize * imyproblemSize; ii++) {
     if (isnan(pmydata->inout[2 * ii]) || isnan(pmydata->inout[2 * ii + 1])) invalid = 1;
     if (isinf(pmydata->inout[2 * ii]) || isinf(pmydata->inout[2 * ii + 1])) invalid = 1;
   }

   /* if loop was necessary */
   if (numberOfRuns > 1) dtime = dtime / numberOfRuns;

   /* calculate the used FLOPS */
   flops = (double)(5.0 * imyproblemSize * imyproblemSize * imyproblemSize * (log2(1.0 * imyproblemSize * imyproblemSize * imyproblemSize)) / dtime);

   /* store the FLOPS in results[1] */
   if (invalid == 1) dresults[1] = INVALID_MEASUREMENT;
     else dresults[1] = flops;

   status = DftiFreeDescriptor(&pmydata->my_desc_handle);

   /* free data */
   free(pmydata->inout);


   /*** out of place run ***/

   /* malloc */
   pmydata->in = (float*)malloc(sizeof(float) * imyproblemSize * imyproblemSize * imyproblemSize * 2);
   pmydata->out = (float*)malloc(sizeof(float) * imyproblemSize * imyproblemSize * imyproblemSize * 2);

   /* create FFT plan */
   status = DftiCreateDescriptor(&pmydata->my_desc_handle, DFTI_SINGLE, DFTI_COMPLEX, 3, len);
   status = DftiSetValue(pmydata->my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
   status = DftiCommitDescriptor(pmydata->my_desc_handle);

   /* init stuff */
   initData_oop(pmydata, imyproblemSize);

   numberOfRuns = 1;

   dstart = bi_gettime();
   /* fft calculation */
   status = DftiComputeForward(pmydata->my_desc_handle, pmydata->in, pmydata->out);
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
        status = DftiComputeForward(pmydata->my_desc_handle, pmydata->in, pmydata->out);
     }
     dend = bi_gettime();

     /* calculate the used time*/
     dtime = dend - dstart;
     dtime -= dTimerOverhead;
   }

   /* if loop was necessary */
   if (numberOfRuns > 1) dtime = dtime / numberOfRuns;

   /* check for overflows */
   for (ii = 0; ii < imyproblemSize * imyproblemSize * imyproblemSize; ii++) {
     if (isnan(pmydata->out[2 * ii]) || isnan(pmydata->out[2 * ii + 1])) invalid = 1;
     if (isinf(pmydata->out[2 * ii]) || isinf(pmydata->out[2 * ii + 1])) invalid = 1;
   }

   /* calculate the used FLOPS */
   flops = (double)(5.0 * imyproblemSize * imyproblemSize * imyproblemSize * (log2(1.0 * imyproblemSize * imyproblemSize * imyproblemSize)) / dtime);

   /* store the FLOPS in results[2] */
   if (invalid == 1) dresults[2] = INVALID_MEASUREMENT;
     else dresults[2] = flops;

   status = DftiFreeDescriptor(&pmydata->my_desc_handle);

   /* free data */
   free(pmydata->in);
   free(pmydata->out);

   return 0;
}


void bi_cleanup(void* mdpv)
{
   mydata_t* pmydata = (mydata_t*)mdpv;
   if (pmydata) {
   free(pmydata->problemSizes);
      free(pmydata);
   }
   return;
}



