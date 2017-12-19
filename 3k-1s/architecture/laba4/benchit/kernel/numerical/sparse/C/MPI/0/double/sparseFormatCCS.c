/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatCCS.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/double/sparseFormatCCS.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "sparseFormatCCS.h"
#include "vector.h"

/*
* convert the part of the matrix on the processor into the CCS format
*/
CCS * MPI_convertToCcsSparse(DT ** matrix, int m, int n) {
	int i, j, _m;
	int c_values=0, c_ccPtr=0;
	CCS * ccs;
	int *not_0_elem, *tempCcPtr;

	DT * ccValues;
	int * rowInd;
	int * ccPtr;

	IDL(INFO, printf("\n---->Entered convertToCcsSparse() for rank=%i\n", rank));

	ccs = (CCS*)malloc(sizeof(CCS));

	not_0_elem = (int*)malloc(n * sizeof(int));
	assert(not_0_elem != NULL);
	tempCcPtr = (int*)malloc(n * sizeof(int));
	assert(tempCcPtr != NULL);

	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	/* count all not-0-elements of the matrixpart per column */
	for(j=0; j<n; j++) {
		not_0_elem[j] = 0;
	}
	for(i=0; i<_m; i++) {
		for(j=0; j<n; j++) {
			if(matrix[i][j] != 0) {
				not_0_elem[j] += 1;
				c_values++;
			}
		}
	}

	if(c_values == 0) {
		ccValues = NULL;
		rowInd = NULL;
		ccPtr = NULL;
	} else {
		c_ccPtr = n;
		ccValues = (DT*)malloc(c_values*sizeof(DT));
		rowInd = (int*)malloc(c_values*sizeof(int));
		ccPtr = (int*)malloc((c_ccPtr+1)*sizeof(int));

		ccPtr[0] = 0;
		tempCcPtr[0] = 0;
		for(i=1; i<n; i++) {
			ccPtr[i] = ccPtr[i-1] + not_0_elem[i-1];
			tempCcPtr[i] = ccPtr[i];
		}
		ccPtr[n] = c_values;

		for(i=0; i<_m; i++) {
			for(j=0; j<n; j++) {
				if(matrix[i][j] != 0) {
					ccValues[tempCcPtr[j]] = matrix[i][j];
					rowInd[tempCcPtr[j]] = i;
					tempCcPtr[j]++;
				}
			}
		}
	}
	
	ccs->values = ccValues;
	ccs->sizeOfValues = c_values;
	ccs->rowIndex = rowInd;
	ccs->sizeOfRowIndex = c_values;
	ccs->ccPointer = ccPtr;
	ccs->sizeOfccPointer = c_ccPtr+1;

	free(not_0_elem);
	free(tempCcPtr);

	IDL(INFO, printf("\n<----Exit convertToCcsSparse() for rank=%i\n", rank));

	return ccs;
}

/*
* root generate the complete CCS be get the parts of the processors
* in this operation the CCS part on the processors (inclusive root) will not be deallocated
*/
CCS * MPI_gatherCCS(CCS * ccs, int m, int n) {
	int i, j, _rank, pos;
	DT * tempValues=NULL, * node_values=NULL;
	int * tempRowInd=NULL, * node_rowInd=NULL;
	int * tempCCPtr=NULL, * node_ccPtr=NULL;
	int * tempPosVal=NULL;
	CCS * tempCCS=NULL;
	MPI_Status status;

	IDL(INFO, printf("\n---->Entered MPI_gatherCCS() for rank=%i\n", rank));

	tempCCS = (CCS*)malloc(sizeof(CCS));

	if(rank==0) {
		tempCCPtr = (int*)calloc((*ccs).sizeOfccPointer, sizeof(int));
		node_ccPtr = (int*)malloc((*ccs).sizeOfccPointer * sizeof(int));

		for(i=0; i<(*ccs).sizeOfccPointer; i++) {
			tempCCPtr[i] = (*ccs).ccPointer[i];
		}

		for(_rank=1; _rank<size; _rank++) {
			MPI_Recv(node_ccPtr, (*ccs).sizeOfccPointer, MPI_iDT, _rank, tag, mycomm, &status);
			for(i=0; i<(*ccs).sizeOfccPointer; i++) {
				tempCCPtr[i] += node_ccPtr[i];
			}
		}
	} else {
		MPI_Send((*ccs).ccPointer, (*ccs).sizeOfccPointer, MPI_iDT, 0, tag, mycomm);
	}
	
	if(rank==0) {
		tempValues = (DT*)malloc(tempCCPtr[(*ccs).sizeOfccPointer-1] * sizeof(DT));
		tempRowInd = (int*)malloc(tempCCPtr[(*ccs).sizeOfccPointer-1] * sizeof(int));

		tempPosVal = (int*)malloc((*ccs).sizeOfccPointer * sizeof(int));
		MPI_copyVector_iDT(tempCCPtr, tempPosVal, (*ccs).sizeOfccPointer);
		for(i=0; i<(*ccs).sizeOfccPointer-1; i++) {
			for(j=(*ccs).ccPointer[i]; j<(*ccs).ccPointer[i+1]; j++) {
				tempValues[ tempPosVal[i] ] = (*ccs).values[j];
				tempRowInd[ tempPosVal[i] ] = (*ccs).rowIndex[j];
				tempPosVal[i]++;
			}
		}

		for(_rank=1; _rank<size; _rank++) {
			MPI_Recv(node_ccPtr, (*ccs).sizeOfccPointer, MPI_iDT, _rank, tag, mycomm, &status);

			node_values = (DT*)malloc(node_ccPtr[(*ccs).sizeOfccPointer-1] * sizeof(DT));
			node_rowInd = (int*)malloc(node_ccPtr[(*ccs).sizeOfccPointer-1] * sizeof(int));

			MPI_Recv(node_values, node_ccPtr[(*ccs).sizeOfccPointer-1], MPI_DT, _rank, tag, mycomm, &status);
				//printf("\n!!! _rank=%i node_ccPtr[%i]=%i node_values[24]=%i",_rank,(*ccs).sizeOfccPointer-1,node_ccPtr[(*ccs).sizeOfccPointer-1],node_values[24]);fflush(stdout);fflush(stderr);
				//for(i=0; i<(*ccs).sizeOfccPointer-1; i++) {
				//	for(j=node_ccPtr[i]; j<node_ccPtr[i+1]; j++) {
				//		printf("\n_rank=%i node_values[%i]=%i",_rank,j,node_values[j]);fflush(stdout);fflush(stderr);
				//	}
				//}
			MPI_Recv(node_rowInd, node_ccPtr[(*ccs).sizeOfccPointer-1], MPI_iDT, _rank, tag, mycomm, &status);
				//printf("\n_rank=%i node_ccPtr[%i]=%i",_rank,(*ccs).sizeOfccPointer-1,node_ccPtr[(*ccs).sizeOfccPointer-1]);fflush(stdout);fflush(stderr);

			pos = _rank * ceil(m / (float)size);
			for(i=0; i<(*ccs).sizeOfccPointer-1; i++) {
				for(j=node_ccPtr[i]; j<node_ccPtr[i+1]; j++) {
					tempValues[ tempPosVal[i] ] = node_values[j];
						//printf("\n_rank=%i node_values[%i]=%i",_rank,j,node_values[j]);fflush(stdout);fflush(stderr);
					tempRowInd[ tempPosVal[i] ] = pos + node_rowInd[j];
					tempPosVal[i]++;
				}
			}
			
			free(node_values);
			free(node_rowInd);
		}
		free(node_ccPtr);
		free(tempPosVal);
	} else {
		MPI_Send((*ccs).ccPointer, (*ccs).sizeOfccPointer, MPI_iDT, 0, tag, mycomm);
		MPI_Send((*ccs).values, (*ccs).ccPointer[(*ccs).sizeOfccPointer-1], MPI_DT, 0, tag, mycomm);
			//printf("\n??? rank=%i (*ccs).ccPointer[%i]=%i (*ccs).values[24]=%i",rank,(*ccs).sizeOfccPointer-1,(*ccs).ccPointer[(*ccs).sizeOfccPointer-1],(*ccs).values[24]);fflush(stdout);fflush(stderr);
		MPI_Send((*ccs).rowIndex, (*ccs).ccPointer[(*ccs).sizeOfccPointer-1], MPI_iDT, 0, tag, mycomm);
	}

	IDL(INFO, printf("\n<----Exit MPI_gatherCCS() for rank=%i\n", rank));

	if(rank == 0) {
		tempCCS->values = tempValues;
		tempCCS->sizeOfValues = tempCCPtr[(*ccs).sizeOfccPointer-1];
		tempCCS->rowIndex = tempRowInd;
		tempCCS->sizeOfRowIndex = tempCCPtr[(*ccs).sizeOfccPointer-1];
		tempCCS->ccPointer = tempCCPtr;
		tempCCS->sizeOfccPointer = (*ccs).sizeOfccPointer;
	} else {
		/* return for all processors excluding root an "empty" CCS */
		tempCCS->values = (DT*)malloc(sizeof(DT));
		tempCCS->sizeOfValues = -1;
		tempCCS->rowIndex = (int*)malloc(sizeof(int));
		tempCCS->sizeOfRowIndex = -1;
		tempCCS->ccPointer = (int*)malloc(sizeof(int));
		tempCCS->sizeOfccPointer = -1;
	}

	return tempCCS;
}

/*
* root generate out of the complete CCS the parts for all processors
* in this operation the complete CCS part on root will be not deallocated
* (inverse operation of MPI_gatherCCS)
*/
CCS * MPI_scatterCCS(CCS * ccs, int m, int n) {
	int pos=0, pos1=0, pos2=0, ind1=0, ind2=0, _m=0, count=0, _rank=0, i=0;
	DT * node_values=NULL;
	int * node_rowInd=NULL, * node_ccPtr=NULL;
	CCS * tempCCS=NULL;
	MPI_Status status;

	IDL(INFO, printf("\n---->Entered MPI_scatterCCS() for rank=%i\n", rank));

	tempCCS = (CCS*)malloc(sizeof(CCS));
	
	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	if(rank==0) {
		for(_rank=0; _rank<size; _rank++) {
			pos1 = _rank * _m;
			if(_rank!=size-1) {
				pos2 = (_rank+1) * _m;
			} else {
				pos2 = m;
			}
				//printf("\n_rank=%i pos1=%i pos2=%i _m=%i",_rank,pos1,pos2,_m);fflush(stdout);fflush(stderr);

			/* count values/indices which will be send to the processor _rank */
			for(i=0, count=0; i<(*ccs).sizeOfValues; i++) {
				if((*ccs).rowIndex[i]>=pos1 && (*ccs).rowIndex[i]<pos2) {
					count++;
				}
			}
				//printf("\n_rank=%i count=%i",_rank,count);fflush(stdout);fflush(stderr);

			/* read out the values/indices which will be send */
			node_values = (DT*)malloc(count * sizeof(DT));
			node_rowInd = (int*)malloc(count * sizeof(int));
			node_ccPtr = (int*)calloc((n+1), sizeof(int));
			pos = _rank * ceil(m / (float)size);
			for(i=0, ind1=0, ind2=0; i<(*ccs).sizeOfValues; i++) {
				while(i >= (*ccs).ccPointer[ind2]) {
					ind2++;
					node_ccPtr[ind2] = node_ccPtr[ind2-1];
				}
				if((*ccs).rowIndex[i]>=pos1 && (*ccs).rowIndex[i]<pos2) {
					node_values[ind1] = (*ccs).values[i];
					node_rowInd[ind1] = (*ccs).rowIndex[i] - pos;
					//printf("\n_rank=%i i=%i (*ccs).values[i]=%i  |  ind1=%i node_values[ind1]=%i",_rank,i,(*ccs).values[i],ind1,node_values[ind1]);fflush(stdout);fflush(stderr);
					ind1++;
					node_ccPtr[ind2]++;
				}
			}

			/* make the send operation for the processor _rank */
			if(_rank==0) {
					//printf("\n_rank=%i ind1=%i",_rank,ind1);fflush(stdout);fflush(stderr);
				tempCCS->values = node_values;
				tempCCS->sizeOfValues = ind1;
				tempCCS->rowIndex = node_rowInd;
				tempCCS->sizeOfRowIndex = ind1;
				tempCCS->ccPointer = node_ccPtr;
				tempCCS->sizeOfccPointer = n+1;
			} else {
					//printf("\n_rank=%i ind1=%i",_rank,ind1);fflush(stdout);fflush(stderr);
				MPI_Send(&ind1, 1, MPI_iDT, _rank, tag, mycomm);
				MPI_Send(node_values, ind1, MPI_DT, _rank, tag, mycomm);
				MPI_Send(node_rowInd, ind1, MPI_iDT, _rank, tag, mycomm);
				MPI_Send(node_ccPtr, n+1, MPI_iDT, _rank, tag, mycomm);
				free(node_values);
				free(node_rowInd);
				free(node_ccPtr);
			}
		}
	} else {
		MPI_Recv(&ind1, 1, MPI_iDT, 0, tag, mycomm, &status);
		tempCCS->values = (DT*)malloc(ind1 * sizeof(DT));
		tempCCS->rowIndex = (int*)malloc(ind1 * sizeof(int));
		tempCCS->ccPointer = (int*)malloc((n+1) * sizeof(int));
		MPI_Recv(tempCCS->values, ind1, MPI_DT, 0, tag, mycomm, &status);
		MPI_Recv(tempCCS->rowIndex, ind1, MPI_iDT, 0, tag, mycomm, &status);
		MPI_Recv(tempCCS->ccPointer, n+1, MPI_iDT, 0, tag, mycomm, &status);
		tempCCS->sizeOfValues = ind1;
		tempCCS->sizeOfRowIndex = tempCCS->sizeOfValues;
		tempCCS->sizeOfccPointer = n+1;
	}

	//if(rank==0) MPI_clearCcsSparse(ccs);

	IDL(INFO, printf("\n<----Exit MPI_scatterCCS() for rank=%i\n", rank));

	return tempCCS;
}

/*
* make a multiplication of the CCS part on the processor and the vector
*/
DT * MPI_CCSxVec(CCS * ccs, int sizeOfB, DT * vec, int sizeOfVec) {
	int i, j, index, m, _m;
	int * recvcounts=NULL, * displs=NULL;
	DT * b;

	IDL(INFO, printf("\n---->Entered MPI_CCSxVec() for rank=%i\n", rank));

	m = sizeOfB;
	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	index = 0;
	if(rank==0) {
		b = (DT*)calloc(m, sizeof(DT));
	} else {
		b = (DT*)calloc(_m, sizeof(DT));
	}

	/* root shares the vector with the others */
	MPI_Bcast(vec, sizeOfVec, MPI_DT, 0, mycomm);
		
	for(j=0; j<(*ccs).sizeOfccPointer-1; j++) {
		for(i=(*ccs).ccPointer[j]; i<(*ccs).ccPointer[j+1]; i++) {
			b[(*ccs).rowIndex[i]] += (*ccs).values[i] * vec[index];
		}
		index++;
	}

	if(rank==0) {
		recvcounts = (int*)malloc(size * sizeof(int));
		displs = (int*)malloc(size * sizeof(int));
		for(i=0; i<size-1; i++) recvcounts[i] = _m;
		recvcounts[size-1] = m - _m * (size-1);
		for(i=0; i<size; i++) displs[i] = i * _m;
	}

	/* root combine the result of each processor */
	MPI_Gatherv(b, _m, MPI_DT, b, recvcounts, displs, MPI_DT, 0, mycomm);
	
	IDL(INFO, printf("\n<----Exit CCSxVec() for rank=%i\n", rank));

	if(rank==0) {
		free(recvcounts);
		free(displs);
		return b;
	} else {
		free(b);
		return NULL;
	}
}

void MPI_printCcsSparse(CCS * ccs) {
	int i;

//	if(rank==0) {
		IDL(INFO, printf("\n---->Entered printCcsSparse() for rank=%i\n", rank));

		printf("\nValues: ");
		for(i=0; i<(*ccs).sizeOfValues; i++) {
			printf(FORMAT1,(*ccs).values[i]);
		}
		printf("\nsizeOfValues: %i\n",(*ccs).sizeOfValues);
	
		printf("\nrowIndex: ");
		for(i=0; i<(*ccs).sizeOfRowIndex; i++) {
			printf("%i, ",(*ccs).rowIndex[i]);
		}
		printf("\nsizeOfRowIndex: %i\n",(*ccs).sizeOfRowIndex);
	
		printf("\nccPointer: ");
		for(i=0; i<(*ccs).sizeOfccPointer; i++) {
			printf("%i, ",(*ccs).ccPointer[i]);
		}
		printf("\nsizeOfccPointer: %i\n",(*ccs).sizeOfccPointer);
	
		IDL(INFO, printf("<----Exit printCcsSparse() for rank=%i\n", rank));
//	}
}

/*
* deallocate the memory for the arrays of the CCS format
*/
void MPI_clearCcsSparse(CCS * ccs) {
	IDL(INFO, printf("\n---->Entered clearCcsSparse() for rank=%i\n", rank));

	free((*ccs).values);
	free((*ccs).rowIndex);
	free((*ccs).ccPointer);
	free(ccs);

	IDL(INFO, printf("\n<----Exit clearCcsSparse() for rank=%i\n", rank));
}


