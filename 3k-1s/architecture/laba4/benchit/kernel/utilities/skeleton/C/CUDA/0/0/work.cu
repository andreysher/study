/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: cuda kernel skeleton
 *******************************************************************/

#include "work.h"
#include "interface.h"

__global__ void moveGPU(int *a, int *b, int max){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if (idx<max) b[idx]=b[idx];
}

void moveCPU(int *a, int *b, int max){
	for(int i=0;i<max;i++)
		b[i]=a[i];
}
