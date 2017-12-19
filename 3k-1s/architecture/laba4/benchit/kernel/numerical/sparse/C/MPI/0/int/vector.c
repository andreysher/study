/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: vector.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/int/vector.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "vector.h"

/*
* create an empty vector
*/
DT * MPI_createVector(int size) {
	DT * vec;

	IDL(INFO, printf("\n---->Entered MPI_createVector() for rank=%i\n", rank));	
	
	vec = (DT*)calloc(size, sizeof(DT));

	IDL(INFO, printf("\n<----Exit MPI_createVector() for rank=%i\n", rank));

	return vec;
}

/*
* create an empty vector
* (with the datatype "int")
*/
int * MPI_createVector_iDT(int size) {
	int * vec;

	IDL(INFO, printf("\n---->Entered MPI_createVector_iDT() for rank=%i\n", rank));	
	
	vec = (int*)calloc(size, sizeof(int));

	IDL(INFO, printf("\n<----Exit MPI_createVector_iDT() for rank=%i\n", rank));

	return vec;
}

/*
* copies the elements from the source to the target array
*/
void MPI_copyVector(DT * source, DT * target, int size) {
	int i;

	IDL(INFO, printf("\n---->Entered MPI_copyVector() for rank=%i\n", rank));	

	for(i=0; i<size; i++) {
		target[i] = source[i];
	}

	IDL(INFO, printf("\n<----Exit MPI_copyVector() for rank=%i\n", rank));
}

/*
* copies the elements from the source to the target array
* (with the datatype "int")
*/
void MPI_copyVector_iDT(int * source, int * target, int size) {
	int i;

	IDL(INFO, printf("\n---->Entered MPI_copyVector_iDT() for rank=%i\n", rank));	

	for(i=0; i<size; i++) {
		target[i] = source[i];
	}

	IDL(INFO, printf("\n<----Exit MPI_copyVector_iDT() for rank=%i\n", rank));
}

/*
* compare 2 vectors and return the biggest distance in the elements,
* or 0 if the vectors are the same
*/
DT MPI_compareVector(DT * vec1, DT * vec2, int size) {
	int i;
	DT eps;

	IDL(INFO, printf("\n---->Entered MPI_compareVector() for rank=%i\n", rank));	

	eps = 0;
	for(i=0; i<size; i++) {
		if(fabs(vec1[i]-vec2[i]) > eps) eps = fabs(vec1[i]-vec2[i]);
	}

	IDL(INFO, printf("\n<----Exit MPI_compareVector() for rank=%i\n", rank));

	return eps;
}

/*
* deallocate the vector
*/
void MPI_clearVector(DT * vec) {
	IDL(INFO, printf("\n---->Entered MPI_clearVector() for rank=%i\n", rank));
	free(vec);
	IDL(INFO, printf("\n<----Exit MPI_clearVector() for rank=%i\n", rank));
}

/*
* fill the vector with random numbers
*	x in [1, 2, ... , 9] for int/long
*	x in [1.0, ..., 9.0] for float/double
*/
void MPI_initRandomVector(DT * vec, int size) {
	int i;

	IDL(INFO, printf("\n---->Entered MPI_initRandomVector() for rank=%i\n", rank));

	for(i=0; i<size; i++) {
		vec[i] = 1+(DT)(9.0*rand()/(RAND_MAX+1.0));
	}

	IDL(INFO, printf("\n<----Exit MPI_initRandomVector() for rank=%i\n", rank));
}

void MPI_printVector(DT * vec, int size) {
	int i;

	IDL(INFO, printf("\n---->Entered MPI_printVector() for rank=%i\n", rank));

	for(i=0; i<size-1; i++) {
		printf(FORMAT1,vec[i]);
	}
	printf(FORMAT2,vec[size-1]);

	IDL(INFO, printf("\n<----Exit MPI_printVector() for rank=%i\n", rank));
}


