/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "work.h"
#include "tools/repeat.h"

__global__ void testLatency(int its, uint* duration){
	uint start, end;
	unsigned clockOverhead=0;
	
	//Measure clock overhead
	for(int i=0; i<=its; i++){
		if(i == 1) clockOverhead = 0;
		start = clock();
		end = clock();
		clockOverhead += end - start;
	}

	duration[0] = clockOverhead;
}
