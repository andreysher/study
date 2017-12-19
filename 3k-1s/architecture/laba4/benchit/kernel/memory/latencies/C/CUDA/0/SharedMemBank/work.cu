/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "work.h"
#include "tools/repeat.h"

__global__ void testLatency(uint* myArray, int arrayLength, uint address, int its, uint* duration){
  extern __shared__ uint sharedMem[];
  //Fill array
  for(int i=0; i < arrayLength; i++)
  	sharedMem[i] = myArray[i];
  	
	uint start, end;
  
  //Measure latency
	unsigned sumTime = 0;
	uint j = address;
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
