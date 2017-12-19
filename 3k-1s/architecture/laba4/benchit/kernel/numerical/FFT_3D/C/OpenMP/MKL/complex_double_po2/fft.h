/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: fft.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/FFT_3D/C/OpenMP/MKL/complex_double_po2/fft.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: 3D Fast Fourier Transform, Powers of 2,
 * double precision, complex data, MKL
 * (C language), OpenMP version
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

#include <mkl.h>

 /* The data structure that holds all the data.*/
typedef struct mydata
{
   myinttype min;
   myinttype max;
   myinttype steps;
   double* in;
   double* out;
   double* inout;
   DFTI_DESCRIPTOR_HANDLE my_desc_handle;
} mydata_t;

#endif




