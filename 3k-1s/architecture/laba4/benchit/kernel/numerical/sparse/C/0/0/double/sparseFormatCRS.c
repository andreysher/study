/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatCRS.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/0/0/double/sparseFormatCRS.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "sparseFormatCRS.h"

/*
convert a normal matrix into the Compressed Row Storage format
*/
CRS * convertToCrsSparse(DT ** matrix, long m, long n) {
	long i, j;
	long c_values=0, c_crPtr=0, index=0;
	CRS * crs;
	long *not_0_elem;

	DT * crValues;
	long * columnInd;
	long * crPtr;

	crs = (CRS*)malloc(sizeof(CRS));

	not_0_elem = (long*)malloc(m * sizeof(long));
	assert(not_0_elem != NULL);

	/* count all not-0-elements of the matrix per row */
	for(i=0; i<m; i++) {
		not_0_elem[i] = 0;
		for(j=0; j<n; j++) {
			if(matrix[i][j] != 0) {
				not_0_elem[i] += 1;
				c_values++;
			}
		}
	}

	if(c_values == 0) {
		crValues = NULL;
		columnInd = NULL;
		crPtr = NULL;
	} else {
		c_crPtr = m;
		crValues = (DT*)malloc(c_values * sizeof(DT));
		columnInd = (long*)malloc(c_values * sizeof(long));
		crPtr = (long*)malloc((c_crPtr+1) * sizeof(long));

		for(i=0; i<m; i++) {
			for(j=0; j<n; j++) {
				if(matrix[i][j] != 0) {
					crValues[index] = matrix[i][j];
					columnInd[index] = j;
					index++;
				}
			}
		}

		crPtr[0] = 0;
		for(i=1; i<m; i++) {
			crPtr[i] = crPtr[i-1] + not_0_elem[i-1];
		}
		crPtr[m] = c_values;
	}
	
	crs->values = crValues;
	crs->sizeOfValues = c_values;
	crs->columnIndex = columnInd;
	crs->sizeOfColumnIndex = c_values;
	crs->crPointer = crPtr;
	crs->sizeOfcrPointer = c_crPtr + 1;

	free(not_0_elem);

	return crs;
}

/*
multiply a CRS matrix with a full vector
*/
DT * CRSxVec(CRS * crs, DT * vec, long sizeOfVec) {
	long i, j;
	DT * b;

	if((*crs).sizeOfValues == 0) {
		return (DT*)calloc((*crs).sizeOfcrPointer, sizeof(DT));
	}

	b = (DT*)calloc((*crs).sizeOfcrPointer, sizeof(DT));
	for(j=0; j<(*crs).sizeOfcrPointer-1; j++) {
		for(i=(*crs).crPointer[j]; i<(*crs).crPointer[j+1]; i++) {
			b[j] += (*crs).values[i] * vec[(*crs).columnIndex[i]];
		} 
	}
	
	return b;
}

void printCrsSparse(CRS * crs) {
	long i;

	printf("\nValues: ");
	for(i=0; i<(*crs).sizeOfValues; i++) {
		printf("%e, ",(*crs).values[i]);
	}
	printf("\nsizeOfValues: %li\n",(*crs).sizeOfValues);

	printf("\ncolumnIndex: ");
	for(i=0; i<(*crs).sizeOfColumnIndex; i++) {
		printf("%li, ",(*crs).columnIndex[i]);
	}
	printf("\nsizeOfColumnIndex: %li\n",(*crs).sizeOfColumnIndex);

	printf("\ncrPointer: ");
	for(i=0; i<(*crs).sizeOfcrPointer; i++) {
		printf("%li, ",(*crs).crPointer[i]);
	}
	printf("\nsizeOfcrPointer: %li\n",(*crs).sizeOfcrPointer);

}

/*
deallocate the memory for the CRS matrix
*/
void clearCrsSparse(CRS * crs) {
	free((*crs).values);
	free((*crs).columnIndex);
	free((*crs).crPointer);
	free(crs);
}


