#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i error = %i\n",__FILE__,__LINE__, error); return 1;}}
#define CHECK_NULL(op) {if (NULL == op){printf("<%s>:%i operand was NULL\n",__FILE__,__LINE__); return 1;}}

#define PREC_SINGLE

#ifdef PREC_SINGLE
	#define DT float
	#define EPSILON 1.0e-6
#else
	#define DT double
	#define EPSILON 1.0e-15
#endif

#define KERNELDES "sgemm"
#define LEGENDY "FLOPS (sgemm)"

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
 	myinttype m, mm, maxm, maxmm;
	DT *hostData[3];
	DT *devData[3];
} mydata_t;

#endif
