/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "work.h"
#include "tools/repeat.h"

//index+=stride;
#define AC(op) sum+=sharedMem[index];index op;
#define ACn(n, op) repeat##n(AC(op))
//sum+=sharedMem[sum];
#define ACOp(n, cmp, op) 	if(stride== cmp){\
	for(int i=0; i<=its; i++){\
		if(i == 1) sumTime = 0;\
		start=clock();\
		ACn(n, op)\
		end=clock();\
		sumTime += end - start;\
	}\
}


#define MES(n) if(accessCt == n){\
ACOp(n, 0, =index)\
ACOp(n, 1, ++)\
ACOp(n, 2, +=2)\
}

__global__ void testLatencyN(uint* myArray, int arrayLength, int accessCt, int stride, int its, uint* duration){
  extern __shared__ uint sharedMem[];
  //Fill array
  for(int i=0; i < arrayLength; i++)
  	sharedMem[i] = myArray[i];
  	
	uint start, end;

  //Measure latency
	unsigned sumTime = 0;
	uint index = 0;
	uint sum = 0;
	
	MES(1)
	MES(2)
	MES(3)
	MES(4)
	MES(5)
	MES(6)
	MES(7)
	MES(8)
	MES(9)
	MES(10)
	MES(11)
	MES(12)
	MES(13)
	MES(14)
	MES(15)
	MES(16)
	MES(17)
	MES(18)
	MES(19)
	MES(20)
	MES(21)
	MES(22)
	MES(23)
	MES(24)
	MES(25)
	MES(26)
	MES(27)
	MES(28)
	MES(29)
	MES(30)
	MES(31)
	MES(32)
			
	myArray[arrayLength] = sum;
	duration[0] = sumTime;
}
