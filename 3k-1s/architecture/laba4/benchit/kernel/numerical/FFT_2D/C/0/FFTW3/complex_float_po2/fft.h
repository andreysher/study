/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: fft.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/FFT_2D/C/0/FFTW3/complex_float_po2/fft.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: 2D Fast Fourier Transform, Powers of 2,
 * single precision, complex data, FFTW3
 * (C language)
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

#include <fftw3.h>

 /* The data structure that holds all the data.*/
typedef struct mydata
{
   myinttype min;
   myinttype max;
   myinttype steps;
   fftwf_complex* in;
   fftwf_complex* out;
   fftwf_complex* inout;
   fftwf_plan p;
} mydata_t;

#endif




