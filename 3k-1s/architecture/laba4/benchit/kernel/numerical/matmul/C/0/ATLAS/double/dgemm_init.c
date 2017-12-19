/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: dgemm_init.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/0/ATLAS/double/dgemm_init.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix Multiply, BLAS, ATLAS (C)
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "interface.h"
#include "dgemm.h"

void bi_getinfo(bi_info* infostruct) {
   char *p = 0;

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

	infostruct->codesequence=bi_strdup("DGEMM");
	infostruct->xaxistext=bi_strdup("Matrix Size");
	infostruct->num_measurements = infostruct->listsize;
	infostruct->numfunctions=1;
  
	/* allocating memory for y axis texts and properties */
  allocYAxis(infostruct);
  
	infostruct->yaxistexts[0] = bi_strdup ("FLOPS");
	infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
	infostruct->legendtexts[0]=bi_strdup("FLOPS");
	infostruct->base_yaxis[0] = 0;

	infostruct->kerneldescription = bi_strdup("Matrix Multiply, BLAS, ATLAS (C)");
}

void* bi_init(int problemSizemax) {

	fds *myfds;
	long lMaxSize;
  /* calculate real maximum problem size */
  
	lMaxSize = (long)bi_get_list_maxelement();
	lMaxSize = lMaxSize*lMaxSize*sizeof(double);

	IDL(3, printf("Enter init\n"));
	myfds=malloc(sizeof(fds));
	if(myfds==NULL) {
		printf("Allocation of structure myfds failed\n");
		exit(127);
	}

	myfds->feld1=malloc(lMaxSize);
	myfds->feld2=malloc(lMaxSize);
	myfds->feld3=malloc(lMaxSize);

	if((myfds->feld1==NULL) || (myfds->feld2==NULL) || (myfds->feld3==NULL)) {
		printf("\nmalloc (%ld bytes) failed in bi_init()\n",(long) (3.0*lMaxSize)); 
		bi_cleanup(myfds);
		exit(127);
		}
	IDL(3, printf("Alloc done %ld Bytes\n", 3*lMaxSize))
	return (myfds);
}

extern void bi_cleanup(void *mcb) {
	fds *data=mcb;
	IDL(3, printf("cleaning..."));
	if(data!=NULL) {
		IDL(3, printf("1"));
		if(data->feld1!=NULL) {
		free(data->feld1);
		data->feld1=NULL;
	}
	IDL(3, printf("2"));
	if(data->feld2!=NULL) {
		free(data->feld2);
		data->feld2=NULL;
	}
	IDL(3, printf("3"));
	if(data->feld3!=NULL) {
		free(data->feld3);
		data->feld3=NULL;
	}
	IDL(3, printf("4\n"));
	free(data);
	}
}
