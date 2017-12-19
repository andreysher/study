/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: vector.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/0/0/int/vector.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "vector.h"

/*
create an empty vector
*/
DT * createVector(long size) {
	DT * vec;

	vec = (DT*)calloc(size, sizeof(DT));

	return vec;
}

/*
copies the elements from the source to the target array
*/
void copyVector(DT * source, DT * target, long size) {
	long i;

	for(i=0; i<size; i++) {
		target[i] = source[i];
	}
}

/*
compare 2 vectors and return the biggest distance in the elements,
or 0 if the vectors are the same
*/
DT compareVector(DT * vec1, DT * vec2, long size) {
	long i;
	DT eps;

	eps = 0;
	for(i=0; i<size; i++) {
		if(fabs(vec1[i]-vec2[i]) > eps) eps = fabs(vec1[i]-vec2[i]);
	}

	return eps;
}

/*
deallocate the vector
*/
void clearVector(DT * vec) {
	free(vec);
}

/*
fill the vector with random numbers
	x in [1, 2, ... , 9] for long
	x in [1.0, ..., 9.0] for double
*/
void initRandomVector(DT * vec, long size) {
	long i;

	for(i=0; i<size; i++) {
		vec[i] = 1 + (DT)(9.0*rand()/(RAND_MAX+1.0));
	}

}

void printVector(DT * vec, long size) {
	long i;

	printf("\n{");
	for(i=0; i<size-1; i++) {
		printf("%d, ",vec[i]);
	}
	printf("%d}\n",vec[size-1]);

}


