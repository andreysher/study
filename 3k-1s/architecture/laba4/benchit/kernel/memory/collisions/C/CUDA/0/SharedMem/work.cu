/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "work.h"
#include "tools/repeat.h"

__global__ void collisionRead(uint* myArray, int arrayLength, int stride, int its, uint* duration){
  extern __shared__ uint sharedMem[];
  int idx=threadIdx.x;

  //Fill array
  for(int i=idx; i < arrayLength; i+=blockDim.x)
  	sharedMem[i] = myArray[i];
  	
  //Sync just in case you have more than 1 warp
  __syncthreads();
  	
	uint start, end;
  
  //Measure latency
	unsigned sumTime = 0;
	uint j = (idx * stride) % arrayLength;
	for(int i=0; i<=its; i++){
		if(i == 1) sumTime = 0;
		start = clock();
		repeat256(j=sharedMem[j];)
		end = clock();
		sumTime += end - start;
	}
	
	myArray[arrayLength] = j;
	duration[0] = sumTime;
}
