/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: dgemm_entry.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/OpenMP/SUNPERF/double/dgemm_entry.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix Multiply, BLAS, SUNPERF
 *******************************************************************/

#include <sunperf.h>
#include <stdio.h>
#include "dgemm.h"
#include "interface.h"

void init_data(fds *myfds, int size) {
	register int x, y;
	long index;
	for(x=0; x<size; x++)
		for(y=0; y<size; y++){
			index=x*size+y;
			myfds->feld1[index]=30;
			myfds->feld2[index]=0.01;
			myfds->feld3[index]=0.0;
		}
	IDL(5, printf("init_data done\n"));
}


int bi_entry(void *mcb, int problemSize,double *results){
	double one=1.0;
	double time=0, start, stop;
	double nOperations=0;
	unsigned long size;
	char N='N';
	double *f1= ((fds*)mcb)->feld1, *f2=((fds*)mcb)->feld2, *f3=((fds*)mcb)->feld3;
	
	if(results == NULL)
		return -1;
	
	size = (unsigned long)bi_get_list_element(problemSize);
	results[0] = size;
	nOperations = (1.0*size)*(1.0*size)*(2.0*size-1.0);
	
	/* init matrices -> cache-friendly */
	init_data(mcb, size);
	
	/* ************************** */
	start=bi_timer();
	//cblas_
	dgemm('N','N', size, size, size, 1.0, f1, size, f2, size, one, f3, size);
	stop=bi_timer();
	/* ************************** */
	
	time=stop-start - dTimerOverhead;
	if (time < 3*dTimerGranularity)   {
		results[1]=INVALID_MEASUREMENT;
	}
	else {
		results[1]=nOperations/time;
	}
	return 0;
}


