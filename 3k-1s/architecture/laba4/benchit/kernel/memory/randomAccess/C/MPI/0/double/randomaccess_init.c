/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: randomaccess_init.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/randomAccess/C/MPI/0/double/randomaccess_init.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: memory bandwith w. random access
 *******************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "interface.h"

long BENCHIT_KERNEL_RANDOMACCESS_MEMSIZE=10*1024*1024;
long N_ACCESSES;
double* dRand=NULL;
double* a=NULL;
double* b=NULL;
long* j=NULL;

int nMeasurements;
int mpiRank = 0, mpiSize = 1; 

double simple_rand() {
	const double a = 12054.67293087436;
	const double b = 0.3183287916948132;
	static double seed = 0.02;
	long i;
	
	seed = a * seed +  b;
	i = (long) seed;
	seed -= i;
	return seed;
} 

void initArays() {
	int i,numMeas=nMeasurements,temp;
	double tempMax=1.0,actMod=.1,start=0.0;
	
	for(i=0;i<2*N_ACCESSES;i++){
		a[i]=simple_rand();
		b[i]=simple_rand();
	}
 	// start with .1
 	numMeas--;
 	actMod=.1;
 	temp=0;
 	while (numMeas>-1)
 	{
 		dRand[numMeas]=tempMax-temp*actMod;
 		temp++;
 		if (tempMax<temp*actMod+start)
 		{
 			if (tempMax==1.0)
 			{
 				tempMax=0.45;
 				actMod=0.1;
 			}
 			else
 			{
 				actMod=actMod/2.0;
 				tempMax=tempMax/2.0;
 			}
 			temp=0;
 		}
 		numMeas--;
 	}
}

int mylog(int mun);
void getRndTimeParameters(void* mem);
void setParameters();
int playOK(int step, int rank);
double dRndA=0.0, dRndB=0.0;
double dMemFactor=0.0;


void bi_getinfo(bi_info* infostruct) {
	int a, i;
	char buff[80], *p;
	float freq=0;
	setParameters();
	infostruct->kerneldescription = bi_strdup("memory bandwith w. random access (C)");
	infostruct->codesequence=bi_strdup("do I=1,N  checksum += A[I]#");
	infostruct->xaxistext=bi_strdup("randomness ");
	infostruct->num_measurements=nMeasurements;
	infostruct->numfunctions= 1;
	
	/* allocating memory for y axis texts and properties */
  allocYAxis(infostruct);

	for(a=0; a<infostruct->numfunctions; a++) {
		infostruct->yaxistexts[a] = bi_strdup ("bandwidth");
	/*	infostruct->legendtexts[a] = bi_strdup ("");*/
		infostruct->selected_result[a] = SELECT_RESULT_HIGHEST;
		infostruct->base_yaxis[a] =  0;
	}
	infostruct->base_xaxis = 0;

	infostruct->kernel_execs_mpi1 = 1;

	sprintf(buff, "%.1f MB",((double)BENCHIT_KERNEL_RANDOMACCESS_MEMSIZE)/(1024*1024),0);
	infostruct->legendtexts[0] = bi_strdup(buff);
}


void* bi_init(int problemSizemax) {

	IDL(3, printf("Enter init\n"));
	MPI_Comm_rank(MPI_COMM_WORLD,&mpiRank);
	MPI_Comm_size(MPI_COMM_WORLD,&mpiSize);
	printf("\n\n MPI Size= %d\n\n",mpiSize);
	setParameters();
	if (a==NULL)
	  a=(double*)malloc(2*N_ACCESSES*sizeof(double));
	if (b==NULL)
	  b=(double*)malloc(2*N_ACCESSES*sizeof(double));
	if (j==NULL)
	  j=(long*)malloc(N_ACCESSES*sizeof(long));
	if (dRand==NULL)
	  dRand=(double*)malloc(nMeasurements*sizeof(double));
	initArays();
	return NULL;
}
void setParameters()
{
	char buff[80], *p;
	int temp=0,i,numMeas=0;
	double actMod,tempMax=1.0;
 	p = bi_getenv("BENCHIT_KERNEL_RANDOMACCESS_MEASUREMENTS",0);
 	if (p==NULL)
 	{
 	  printf("BENCHIT_KERNEL_RANDOMACCESS_MEASUREMENTS not set, returning");
 	  exit(127);
 	}
 	numMeas=atoi(p);
 	if (numMeas==0)
 	{
 	  printf("BENCHIT_KERNEL_RANDOMACCESS_MEASUREMENTS is 0, returning");
 	  exit(127);
 	}
 	nMeasurements=numMeas;
 	
 	
 	// decrease by 1/2 of old value
 	p = bi_getenv("BENCHIT_KERNEL_RANDOMACCESS_MEMSIZE",0);
 	if (p==NULL)
 	{
 	  printf("BENCHIT_KERNEL_RANDOMACCESS_MEMSIZE not set, returning");
 	  exit(127);
 	}
  BENCHIT_KERNEL_RANDOMACCESS_MEMSIZE=atoi(p);
 	N_ACCESSES=(long)(atoi(p)/(4*sizeof(double) + sizeof(long)));
 	if (N_ACCESSES==0)
 	{
 	  printf("BENCHIT_KERNEL_RANDOMACCESS_MEMSIZE is 0, returning");
 	  exit(127);
 	}
}

extern void bi_cleanup(void *mem) {
	IDL(3, printf("cleaning..."));
	if(mem!=NULL) {
		free(mem);
	}
}
