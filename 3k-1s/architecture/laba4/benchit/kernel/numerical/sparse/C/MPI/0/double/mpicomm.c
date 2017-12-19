/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: mpicomm.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/double/mpicomm.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "mpicomm.h"

/*
* this function creates a new communicator for the program, thru that the program can handle processors with a empty matrix
* and the workload for the processors is better shared 
*/
void createMyMPIComm(int m) {
	int num_ranks=0, i=0, _m=0;
	int * ranks=NULL;

	IDL(INFO, printf("\n---->Entered createMyMPIComm() for globalrank=%i\n", globalrank));

	size = globalsize;
	ranks = (int*)calloc(globalsize, sizeof(int));
	for(i=0; i<globalsize; i++) {
		_m = ceil(m / (float)size);
		_m = m - _m * (size-1);
		if(_m < 1) {
			size--;
			ranks[i] = size;
			num_ranks++;
		} else break;
	}
	//if(num_ranks!=0) if(globalrank==0) IDL(0, printf("\nm=%i, numr=%i",m,num_ranks));

	/*
	IDL(INFO, printf("\n%i\n",size));
	for(i=0; i<globalsize; i++) {
		IDL(INFO, printf("%i, ",ranks[i]));
	} */

	MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);
	MPI_Group_excl(MPI_GROUP_WORLD, num_ranks, ranks, &mygroup)  ;
	MPI_Comm_create(MPI_COMM_WORLD, mygroup, &mycomm);

	MPI_Group_rank(mygroup,&rank);

	IDL(INFO, printf("\n<----Exit createMyMPIComm() for globalrank=%i\n", globalrank));
}

/*
* clean up the created communicator
*/
void freeMyMPIComm() {
	IDL(INFO, printf("\n---->Entered freeMyMPIComm() for globalrank=%i\n", globalrank));

	if(rank != MPI_UNDEFINED) MPI_Comm_free(&mycomm) ;
	MPI_Group_free(&MPI_GROUP_WORLD);
	MPI_Group_free(&mygroup);

	IDL(INFO, printf("\n<----Exit freeMyMPIComm() for globalrank=%i\n", globalrank));
}


