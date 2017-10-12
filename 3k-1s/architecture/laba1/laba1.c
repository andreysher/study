#include <stdio.h>
#include "rdtsc.h"
#include <stdlib.h>
#define SIZE_OF_ARR 1000000
#define ITERATIONS 2000000000

int main(int argc, char const *argv[])
{  
  	
    unsigned long long start;
    unsigned long long finish;  
	
	long i;
	double a = 1.0;
	double b = 1.0;
	start = rdtsc();
	for(i = 0; i < ITERATIONS; ++i){
		a = a + b;
	}
	finish = rdtsc();
	double res = (double)(finish - start)/ITERATIONS;
	printf("%lf\n", res);
	if(a > ITERATIONS){
		printf("yes\n");
	}
	return 0;
}
