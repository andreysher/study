/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: dgemm_init.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/OpenMP/MKL/double-optimized/dgemm_init.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix Multiply, BLAS, MKL (C) - OpenMP version
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "interface.h"
#include <omp.h>
#include "dgemm.h"

void bi_getinfo(bi_info* infostruct) {
   char *p = 0;

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

	infostruct->kerneldescription = bi_strdup("Matrix Multiply, BLAS, MKL (C) - OpenMP version for SGI Altix");	
	infostruct->codesequence=bi_strdup("DGEMM");
	infostruct->xaxistext=bi_strdup("Matrix Size");
	infostruct->num_measurements = infostruct->listsize;
  
	/* allocating memory for y axis texts and properties */
  allocYAxis(infostruct);
  
	infostruct->yaxistexts[0] = bi_strdup ("FLOPS");
	infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
	infostruct->legendtexts[0]=bi_strdup("FLOPS");
	infostruct->base_yaxis[0] = 0;

	infostruct->num_processes = 1;
	infostruct->num_threads_per_process = atol(bi_getenv("BENCHIT_NUM_CPUS",1));
	infostruct->kernel_execs_mpi1 = 0;
	infostruct->kernel_execs_mpi2 = 0;
	infostruct->kernel_execs_pvm = 0;
	infostruct->kernel_execs_omp = 1;
	infostruct->kernel_execs_pthreads = 0;
	infostruct->numfunctions = 1;
}

void* bi_init(int problemSizemax) {

	fds *myfds;
	long lMaxSize;

	IDL(3, printf("Enter init\n"));
	myfds=malloc(sizeof(fds));
	if(myfds==NULL) {
		printf("Allocation of structure myfds failed\n");
		exit(127);
	}

	return (myfds);
}

extern void bi_cleanup(void *mcb) {
	fds *data=mcb;
	free(data);
}
