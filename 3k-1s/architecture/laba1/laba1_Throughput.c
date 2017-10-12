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
	printf("%llu\n", finish - start);
	if(a > ITERATIONS){
		printf("yes\n");
	}
	return 0;
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
	printf("%llu\n", finish - start);
	if(a > ITERATIONS){
		printf("yes\n");
	}
	return 0;
}