/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/FFT_3D/C/0/FFTW3/complex_double_npo2/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: 3D Fast Fourier Transform, Non-Powers of 2,
 * double precision, complex data, FFTW3
 * (C language)
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <interface.h>
#include <fft.h>

int functionCount;
int valuesPerFunction;


void initData_ip(mydata_t* mdpv,int problemSize)
{
   int i;
   for (i = 0; i < problemSize * problemSize * problemSize; i++)
   {
      mdpv->inout[i][0] = 0.0;
      mdpv->inout[i][1] = 0.0;
   }
}


void initData_oop(mydata_t* mdpv,int problemSize)
{
   int i;
   for (i = 0; i < problemSize * problemSize * problemSize; i++)
   {
      mdpv->in[i][0] = 1.1/(i+1);
      mdpv->in[i][1] = 1.2/(i+1);
      mdpv->out[i][0] = 0.0;
      mdpv->out[i][1] = 0.0;
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


   penv = (mydata_t*)malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   evaluate_environment(penv);
   infostruct->codesequence = bi_strdup("fftw_execute()");
   infostruct->kerneldescription = bi_strdup("3D Fast Fourier Transform, Non-Powers of 2, double precision, complex data, FFTW3 (C)");
   infostruct->xaxistext = bi_strdup("Problem Size");
   infostruct->num_measurements = penv->steps;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
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

   /* calculate real problemSize */
   imyproblemSize = (int)(pmydata->problemSizes[iproblemSize - 1]);

   /* store the value for the x axis in results[0] */
   dresults[0] = (double)imyproblemSize;


   /*** in place run ***/

   /* malloc */
   pmydata->inout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * imyproblemSize * imyproblemSize * imyproblemSize);

   /* create FFT plan */
   pmydata->p = fftw_plan_dft_3d(imyproblemSize, imyproblemSize, imyproblemSize, pmydata->inout, pmydata->inout, FFTW_FORWARD, FFTW_ESTIMATE);


   /* init stuff */
   initData_ip(pmydata, imyproblemSize);

   numberOfRuns = 1;

   dstart = bi_gettime();
   /* fft calculation */
   fftw_execute(pmydata->p);
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
       fftw_execute(pmydata->p);
     }
     dend = bi_gettime();

     dtime = dend - dstart;
     dtime -= dTimerOverhead;
   }

   /* check for overflows */
   for (ii = 0; ii < imyproblemSize * imyproblemSize * imyproblemSize; ii++) {
     if (isnan(pmydata->inout[ii][0]) || isnan(pmydata->inout[ii][1])) invalid = 1;
     if (isinf(pmydata->inout[ii][0]) || isinf(pmydata->inout[ii][1])) invalid = 1;
   }

   /* if loop was necessary */
   if (numberOfRuns > 1) dtime = dtime / numberOfRuns;

   /* calculate the used FLOPS */
   flops = (double)(5.0 * imyproblemSize * imyproblemSize * imyproblemSize * (log2(1.0 * imyproblemSize * imyproblemSize * imyproblemSize)) / dtime);

   /* store the FLOPS in results[1] */
   if (invalid == 1) dresults[1] = INVALID_MEASUREMENT;
     else dresults[1] = flops;

   fftw_destroy_plan(pmydata->p);

   /* free data */
   fftw_free(pmydata->inout);


   /*** out of place run ***/

   /* malloc */
   pmydata->in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * imyproblemSize * imyproblemSize * imyproblemSize);
   pmydata->out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * imyproblemSize * imyproblemSize * imyproblemSize);

   /* create FFT plan */
   pmydata->p = fftw_plan_dft_3d(imyproblemSize, imyproblemSize, imyproblemSize, pmydata->in, pmydata->out, FFTW_FORWARD, FFTW_ESTIMATE);

   /* init stuff */
   initData_oop(pmydata, imyproblemSize);

   numberOfRuns = 1;

   dstart = bi_gettime();
   /* fft calculation */
   fftw_execute(pmydata->p);
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
        fftw_execute(pmydata->p);
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
     if (isnan(pmydata->out[ii][0]) || isnan(pmydata->out[ii][1])) invalid = 1;
     if (isinf(pmydata->out[ii][0]) || isinf(pmydata->out[ii][1])) invalid = 1;
   }

   /* calculate the used FLOPS */
   flops = (double)(5.0 * imyproblemSize * imyproblemSize * imyproblemSize * (log2(1.0 * imyproblemSize * imyproblemSize * imyproblemSize)) / dtime);

   /* store the FLOPS in results[2] */
   if (invalid == 1) dresults[2] = INVALID_MEASUREMENT;
     else dresults[2] = flops;

   fftw_destroy_plan(pmydata->p);

   /* free data */
   fftw_free(pmydata->in);
   fftw_free(pmydata->out);

   return 0;
}


void bi_cleanup(void* mdpv)
{
   mydata_t* pmydata = (mydata_t*)mdpv;
   if (pmydata) {
   fftw_free(pmydata->problemSizes);
      free(pmydata);
   }
   return;
}



