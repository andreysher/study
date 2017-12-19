/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: dgemm_entry.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/OpenMP/MKL/double-optimized/dgemm_entry.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix Multiply, BLAS, MKL (C) - OpenMP version
 *******************************************************************/

#include <mkl_cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include "dgemm.h"
#include "interface.h"
#include <omp.h>

void init_data(fds *myfds, int size) {
	long x, index, max;
#pragma omp parallel for schedule(static,1) private(x,index,max) shared(myfds,size)
	for(x = 0; x < size; x++) {
		index = x * size;
		max = index + size;
		for(index; index < max; index++) {
			myfds->feld1[index] = 30.0;
			myfds->feld2[index] = 0.01;
			myfds->feld3[index] = 0.0;
		}
	}
	IDL(5, printf("init_data done\n"));
}


int bi_entry(void *mcb, int problemSize,double *results){
	double one=1.0;
	double time=0, start, stop;
	double nOperations=0.0;
	long lCurrentSize;
	unsigned long size, optsize, diff;
	char N='N';
	double *f1, *f2, *f3;
	int ii, jj, kk, ompnumthreads;
	fds* pmydata;

	pmydata = (fds*)mcb;
        ompnumthreads = omp_get_max_threads();

	if(results == NULL)
		return -1;
	
	size = (unsigned long)bi_get_list_element(problemSize);

	lCurrentSize = size*size*sizeof(double);

	pmydata->feld1=malloc(lCurrentSize);
	pmydata->feld2=malloc(lCurrentSize);
	pmydata->feld3=malloc(lCurrentSize);

	f1=pmydata->feld1; f2=pmydata->feld2; f3=pmydata->feld3;

	if((f1==NULL) || (f2==NULL) || (f3==NULL)) {
		printf("\nmalloc (%ld bytes) failed in bi_entry()\n",(long) (3.0*lCurrentSize)); 
		bi_cleanup(mcb);
		exit(127);
		}

	start=bi_gettime();
	init_data(mcb, size);
	stop=bi_gettime();
	time=stop-start - dTimerOverhead;

	/* ************************** */
	start=bi_gettime();
	diff = size % ompnumthreads;
	optsize = size - diff;

	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, optsize, optsize, size, 1.0, pmydata->feld1, size, pmydata->feld2, size, one, pmydata->feld3, size);

	if (diff > 0)
	{
		/*right part*/
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, optsize, diff, size, 1.0, pmydata->feld1, size, &(pmydata->feld2[optsize]), size, one, &(pmydata->feld3[optsize]), size);
		/*bottom part*/
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, diff, optsize, size, 1.0, &(pmydata->feld1[size*optsize]), size, pmydata->feld2, size, one, &(pmydata->feld3[size*optsize]), size);
		/*bottom right part*/
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, diff, diff, size, 1.0, &(pmydata->feld1[size*optsize]), size, &(pmydata->feld2[optsize]), size, one, &(pmydata->feld3[size*optsize+optsize]), size);
	}

	stop=bi_gettime();

	results[0] = size;
	nOperations = (1.0*size)*(1.0*size)*(2.0*size-1.0);
	time=stop-start - dTimerOverhead;
	if (time < 3*dTimerGranularity)   {
		results[1]=INVALID_MEASUREMENT;
	}
	else {
		results[1]=nOperations/time;
	}

	if(mcb!=NULL) {
		if(f1!=NULL) {
			free(f1);
			f1=NULL;
		}
		if(f2!=NULL) {
			free(f2);
			f2=NULL;
		}
		if(f3!=NULL) {
			free(f3);
			f3=NULL;
		}
	}

	return 0;
}


