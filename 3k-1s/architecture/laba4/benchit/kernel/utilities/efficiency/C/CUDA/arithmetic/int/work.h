/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
 *******************************************************************/

#ifndef __work_h
#define __work_h

#define USE_INT

#include "repeat.h"
#include "instructions.h"

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));fflush(stdout);exit(122);}}
/*start kernel, wait for finish and check errors*/
#define CUDA_CHECK_KERNEL_SYNC(...) __VA_ARGS__;CUDA_CHECK(cudaDeviceSynchronize())
/*only check if kernel start is valid*/
#define CUDA_CHECK_KERNEL(...) __VA_ARGS__;CUDA_CHECK(cudaGetLastError())

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata
{
	uint *h_out, *d_out;
	uint *h_ts, *d_ts;
	int maxThreadsPerBlock;
} mydata_t;

#define EXEC_FUNC(OP) EXEC_FUNC2(OP, TYPE) 
#define EXEC_FUNC2(OP, TYPE) PASTE_TOGETHER(K_, OP, _, TYPE, _DEP128)
#define PASTE_TOGETHER(ONE, TWO, THREE, FOUR, FIVE) ONE ## TWO ## THREE ## FOUR ## FIVE

#endif
