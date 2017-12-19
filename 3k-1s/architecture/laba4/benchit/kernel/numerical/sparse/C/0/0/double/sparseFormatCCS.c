/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatCCS.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/0/0/double/sparseFormatCCS.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "sparseFormatCCS.h"

/*
convert a normal matrix into the Compressed Column Storage format
*/
CCS * convertToCcsSparse(DT ** matrix, long m, long n) {
	long i, j;
	long c_values=0, c_ccPtr=0;
	CCS * ccs;
	long *not_0_elem, *tempCcPtr;

	DT * ccValues;
	long * rowInd;
	long * ccPtr;

	ccs = (CCS*)malloc(sizeof(CCS));

	not_0_elem = (long*)malloc(n * sizeof(long));
	assert(not_0_elem != NULL);
	tempCcPtr = (long*)malloc(n * sizeof(long));
	assert(tempCcPtr != NULL);

	/* count all not-0-elements of the matrix per column */
	for(j=0; j<n; j++) {
		not_0_elem[j] = 0;
	}
	for(i=0; i<m; i++) {
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
		ccValues = (DT*)malloc(c_values * sizeof(DT));
		rowInd = (long*)malloc(c_values * sizeof(long));
		ccPtr = (long*)malloc((c_ccPtr+1) * sizeof(long));

		ccPtr[0] = 0;
		tempCcPtr[0] = 0;
		for(i=1; i<n; i++) {
			ccPtr[i] = ccPtr[i-1] + not_0_elem[i-1];
			tempCcPtr[i] = ccPtr[i];
		}
		ccPtr[n] = c_values;

		for(i=0; i<m; i++) {
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
	ccs->sizeOfccPointer = c_ccPtr + 1;

	free(not_0_elem);
	free(tempCcPtr);

	return ccs;
}

/*
multiply a CCS matrix with a full vector
*/
DT * CCSxVec(CCS * ccs, long sizeOfB, DT * vec, long sizeOfVec) {
	long i, j, index;
	DT * b;

	if((*ccs).sizeOfValues == 0) {
		return (DT*)calloc(sizeOfB, sizeof(DT));
	}

	index = 0;
	b = (DT*)calloc(sizeOfB, sizeof(DT));
	for(j=0; j<(*ccs).sizeOfccPointer-1; j++) {
		for(i=(*ccs).ccPointer[j]; i<(*ccs).ccPointer[j+1]; i++) {
			b[(*ccs).rowIndex[i]] += (*ccs).values[i] * vec[index];
		}
		index++;
	}

	return b;
}

void printCcsSparse(CCS * ccs) {
	long i;

	printf("\nValues: ");
	for(i=0; i<(*ccs).sizeOfValues; i++) {
		printf("%e, ",(*ccs).values[i]);
	}
	printf("\nsizeOfValues: %li\n",(*ccs).sizeOfValues);

	printf("\nrowIndex: ");
	for(i=0; i<(*ccs).sizeOfRowIndex; i++) {
		printf("%li, ",(*ccs).rowIndex[i]);
	}
	printf("\nsizeOfRowIndex: %li\n",(*ccs).sizeOfRowIndex);

	printf("\nccPointer: ");
	for(i=0; i<(*ccs).sizeOfccPointer; i++) {
		printf("%li, ",(*ccs).ccPointer[i]);
	}
	printf("\nsizeOfccPointer: %li\n",(*ccs).sizeOfccPointer);

}

/*
deallocate the memory for the CRS matrix
*/
void clearCcsSparse(CCS * ccs) {
	free((*ccs).values);
	free((*ccs).rowIndex);
	free((*ccs).ccPointer);
	free(ccs);
}


