/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: fft.h 1 2011-01-29 fschmitt $
 * $URL: svn+ssh://benchit@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/FFT_2D/C/CUDA/CUFFT/complex_double_npo2/fft.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: 2D Fast Fourier Transform, Non-Powers of 2,
 * double precision, complex data, CUFFT
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

#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define CUFFT_CHECK(cmd) {cufftResult error = cmd; if(error!=CUFFT_SUCCESS){printf("<%s>:%i error = %i\n",__FILE__,__LINE__, error); return 1;}}
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=CUDA_SUCCESS){printf("<%s>:%i error = %i\n",__FILE__,__LINE__, error); return 1;}}
#define CHECK_NULL(op) {if (NULL == op){printf("<%s>:%i operand was NULL\n",__FILE__,__LINE__); return 1;}}

#define PREC_DOUBLE

#ifdef PREC_SINGLE
#define DT_s float
#define DT float2
#define EPSILON 1.0e-6
#else
#define DT_s double
#define DT double2
#define EPSILON 1.0e-15
#endif

typedef struct
{
    size_t elements, maxElements;
    DT * hostData[2];
    DT * devData[2];
} mydata_t;

#endif
