/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "work.h"

__global__ void getClock(int maxThreads, uint* clocks){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	//if(id < maxThreads)
		clocks[id] = clock();
}
