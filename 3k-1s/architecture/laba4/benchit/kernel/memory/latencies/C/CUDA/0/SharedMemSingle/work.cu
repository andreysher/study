/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "work.h"
#include "tools/repeat.h"

__global__ void testLatency(uint* myArray, int arrayLength, int repeatCt, int its, uint* duration){
  extern __shared__ uint sharedMem[];
  //Fill array
  for(int i=0; i < arrayLength; i++)
  	sharedMem[i] = myArray[i];
  	
	uint start, end;
	/*unsigned clockOverhead=0;
	
	//Measure clock overhead
	for(int i=0; i<=its; i++){
		if(i == 1) clockOverhead = 0;
		start = clock();
		end = clock();
		clockOverhead += end - start;
	}*/
  
  //Measure latency
	unsigned sumTime = 0;
	uint j = 0;
	for(int i=0; i<=its; i++){
		if(i == 1) sumTime = 0;
		start = clock();
		repeat256(j=sharedMem[j];)
		end = clock();
		sumTime += end - start;
	}
		
	myArray[arrayLength] += j;
	duration[0] = sumTime;
	
	j = 0;
	for(int i=0; i<=its; i++){
		if(i == 1) sumTime = 0;
		int k = 0;
		start = clock();
		do{
			k++;
			repeat6(j=sharedMem[j];)
		}while(k<repeatCt);
		end = clock();
		sumTime += end - start;
	}

	myArray[arrayLength] += j;
	duration[1] = sumTime;
}
