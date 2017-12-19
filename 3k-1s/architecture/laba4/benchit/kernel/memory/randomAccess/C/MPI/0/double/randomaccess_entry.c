/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: randomaccess_entry.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/randomAccess/C/MPI/0/double/randomaccess_entry.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: memory bandwith w. random access
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "interface.h"

extern long N_ACCESSES;
extern long BENCHIT_KERNEL_RANDOMACCESS_MEMSIZE;
extern double* a;
extern double* b;
extern long* j;

double r=1.0;

extern int mpiRank, mpiSize;
	
double simple_rand();

extern double* dRand;
	
void mix_index(double randomness) {
	double r;
	long i;
	
	for(i = 0 ; i < N_ACCESSES; i++) {
		if (simple_rand() >= randomness)
			j[i]=i;
		else {
			r = simple_rand() * 2.0 * (double)N_ACCESSES;
			j[i] = (long) r;
		}
	}
}
int bi_entry(void *mcb, int problemSize,double *results){
	int i;
	double start, stop;
	double r = 2.1;
	double randomness; 
	initArays();
	
	randomness = dRand[problemSize-1];
	mix_index(randomness);
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(mpiRank == 0) {
		start = bi_timer();
	}
	MPI_Barrier(MPI_COMM_WORLD);
		
	for(i=0;i<N_ACCESSES;i++)
		r = r + a[j[i]];
		
	MPI_Barrier(MPI_COMM_WORLD);	
	if (mpiRank==0) {
		stop=bi_timer();
	}
	
	if (r < N_ACCESSES/3)
		printf("<");	
	if (results) {
		results[0] = randomness;
		results[1] = ((double)(mpiSize*N_ACCESSES*(1*sizeof(double)+sizeof(long))))/(stop-start);
	}
	return 0;
}

/*
double streaming(){
	int i;
	double start=0.0, stop=1.0E+9;
	double r;
		
	r = 1.1;
	initArays();
	
	MPI_Barrier(MPI_COMM_WORLD);

	if(mpiRank == 0)
		start = MPI_Wtime();
	
	MPI_Barrier(MPI_COMM_WORLD);
			
	for(i=0;i<N_ACCESSES;i++)
		r = r + b[i];
		
	MPI_Barrier(MPI_COMM_WORLD);
	
	if (mpiRank==0)
		stop=MPI_Wtime();
	
	if (r < N_ACCESSES/3)
		printf("<");
		
	return stop-start;
}
*/

