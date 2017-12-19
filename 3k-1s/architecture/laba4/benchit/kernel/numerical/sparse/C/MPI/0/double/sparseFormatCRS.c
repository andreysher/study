/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatCRS.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/double/sparseFormatCRS.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "sparseFormatCRS.h"

/*
* convert the part of the matrix on the processor into the CRS format
*/
CRS * MPI_convertToCrsSparse(DT ** matrix, int m, int n) {
	int i, j, _m;
	int c_values=0, c_crPtr=0, index=0;
	CRS * crs;
	int *not_0_elem;

	DT * crValues;
	int * columnInd;
	int * crPtr;

	IDL(INFO, printf("\n---->Entered MPI_convertToCrsSparse() for rank=%i\n", rank));

	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	crs = (CRS*)malloc(sizeof(CRS));

	not_0_elem = (int*)malloc(_m * sizeof(int));
	assert(not_0_elem != NULL);

	/* count all not-0-elements of the matrixpart per row */
	for(i=0; i<_m; i++) {
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
		c_crPtr = _m;
		crValues = (DT*)malloc(c_values * sizeof(DT));
		columnInd = (int*)malloc(c_values * sizeof(int));
		crPtr = (int*)malloc((c_crPtr+1) * sizeof(int));

		for(i=0; i<_m; i++) {
			for(j=0; j<n; j++) {
				if(matrix[i][j] != 0) {
					crValues[index] = matrix[i][j];
					columnInd[index] = j;
					index++;
				}
			}
		}

		crPtr[0] = 0;
		for(i=1; i<_m; i++) {
			crPtr[i] = crPtr[i-1] + not_0_elem[i-1];
		}
		crPtr[_m] = c_values;
	}
	
	crs->values = crValues;
	crs->sizeOfValues = c_values;
	crs->columnIndex = columnInd;
	crs->sizeOfColumnIndex = c_values;
	crs->crPointer = crPtr;
	crs->sizeOfcrPointer = c_crPtr + 1;

	free(not_0_elem);

	IDL(INFO, printf("\n<----Exit MPI_convertToCrsSparse() for rank=%i\n", rank));

	return crs;
}

/*
* root generate the complete CRS be get the parts of the processors
* in this operation the CRS part on the processors (inclusive root) will not be deallocated
*/
CRS * MPI_gatherCRS(CRS * crs, int m, int n) {
	DT * tempValues=NULL;
	int * tempColumnIndex=NULL, * tempCrPointer=NULL;
	CRS * tempCRS=NULL;
	int * sizeOfvalues_perNode=NULL, * recvcounts=NULL, * displs=NULL;
	int sizeOfTempValues=0, i=0, j=0, _m=0, _rank=0;

	IDL(INFO, printf("\n---->Entered MPI_gatherCRS() for rank=%i\n", rank));

	tempCRS = (CRS*)malloc(sizeof(CRS));

	/* get the number of values/indices of the CRS format on each processor */
	if(rank==0) {
		sizeOfvalues_perNode = (int*)malloc(size*sizeof(int));
	} else {
		sizeOfvalues_perNode = (int*)malloc(sizeof(int));	// without MPI_Gatherv don't work
	}
	MPI_Gather(&(crs->sizeOfValues), 1, MPI_iDT, sizeOfvalues_perNode, 1, MPI_iDT, 0, mycomm);

	/* recieve the values/indices of each processor */
	if(rank==0) {
		displs = (int*)calloc(size, sizeof(int));
		for(i=1; i<size; i++) {
			displs[i] = displs[i-1] + sizeOfvalues_perNode[i-1];
		}
		sizeOfTempValues = displs[size-1] + sizeOfvalues_perNode[size-1];
		
		tempValues = (DT*)malloc(sizeOfTempValues * sizeof(DT));
		tempColumnIndex = (int*)malloc(sizeOfTempValues * sizeof(int));
	}
	MPI_Gatherv(crs->values, crs->sizeOfValues, MPI_DT, tempValues, sizeOfvalues_perNode, displs, MPI_DT, 0, mycomm);
	MPI_Gatherv(crs->columnIndex, crs->sizeOfColumnIndex, MPI_iDT, tempColumnIndex, sizeOfvalues_perNode, displs, MPI_iDT, 0, mycomm);
	if(rank==0) {
		free(sizeOfvalues_perNode);
		free(displs);
	} else {
		free(sizeOfvalues_perNode);
	}

	/* recieve the crPointer of each processor */
	if(rank==0) {
		tempCrPointer = (int*)calloc((m+1), sizeof(int));
		recvcounts = (int*)malloc(size * sizeof(int));
		displs = (int*)malloc(size * sizeof(int));
		_m = ceil(m / (float)size);
		for(i=0; i<size-1; i++) {
			recvcounts[i] = _m;
			displs[i] = i*_m + 1;
		}
		recvcounts[size-1] = m - _m * (size-1);
		displs[size-1] = (size-1)*_m + 1;
		if(size==1) displs[0] = 0;	// without MPI_Gatherv don't work for 1CPU, for that the displacement must be 0, don't know why
	} else {
		recvcounts = (int*)malloc(sizeof(int));
	}
	if(size==1) {
		// without MPI_Gatherv don't work for 1CPU, for that the displacement must be 0, don't know why, because of that I need "&(tempCrPointer[1])"
		MPI_Gatherv(&((crs->crPointer)[1]), (crs->sizeOfcrPointer)-1, MPI_iDT, &(tempCrPointer[1]), recvcounts, displs, MPI_iDT, 0, mycomm);
	} else {
		MPI_Gatherv(&((crs->crPointer)[1]), (crs->sizeOfcrPointer)-1, MPI_iDT, tempCrPointer, recvcounts, displs, MPI_iDT, 0, mycomm);
	}
	if(rank==0) {
		free(recvcounts);
		free(displs);
	} else {
		free(recvcounts);
	}

	/* make a work on the crPointer for the complete CRS */
	if(rank==0) {
		for(_rank=0; _rank<size-1; _rank++) {
			for(j=1+_rank*_m; j<1+(_rank+1)*_m; j++) tempCrPointer[j] += tempCrPointer[_rank*_m]; // tempCrPointer[j] += tempCrPointer[1+_rank*_m-1];
		}
		for(j=1+(size-1)*_m; j<m+1; j++)  tempCrPointer[j] += tempCrPointer[(size-1)*_m]; 		// tempCrPointer[j] += tempCrPointer[1+(size-1)*_m-1];
	}

	IDL(INFO, printf("\n<----Exit MPI_gatherCRS() for rank=%i\n", rank));

	if(rank==0) {
		tempCRS->values = tempValues;
		tempCRS->sizeOfValues = sizeOfTempValues;
		tempCRS->columnIndex = tempColumnIndex;
		tempCRS->sizeOfColumnIndex = sizeOfTempValues;
		tempCRS->crPointer = tempCrPointer;
		tempCRS->sizeOfcrPointer = m + 1;
	} else {
		/* return for all processors excluding root an "empty" CRS */
		tempCRS->values = (DT*)malloc(sizeof(DT));
		tempCRS->sizeOfValues = -1;
		tempCRS->columnIndex = (int*)malloc(sizeof(int));
		tempCRS->sizeOfColumnIndex = -1;
		tempCRS->crPointer = (int*)malloc(sizeof(int));
		tempCRS->sizeOfcrPointer = -1;
	}

	return tempCRS;
}

/*
* root generate out of the complete CRS the parts for all processors
* in this operation the complete CRS part on root will be not deallocated
* (inverse operation of MPI_gatherCRS)
*/
CRS * MPI_scatterCRS(CRS * crs, int m, int n) {
	DT * tempValues=NULL;
	int * tempColumnIndex=NULL, * tempCrPointer=NULL;
	CRS * tempCRS=NULL;
	int _m=0, i=0, first=0;
	int * sendcounts=NULL, * displs=NULL;

	IDL(INFO, printf("\n---->Entered MPI_scatterCRS() for rank=%i\n", rank));

	tempCRS = (CRS*)malloc(sizeof(CRS));

	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	/* send the processor-relevant part of the complete crPointer to each processor */
	if(rank==0) {
		sendcounts = (int*)malloc(size * sizeof(int));
		displs = (int*)malloc(size * sizeof(int));
		for(i=0; i<size-1; i++) {
			sendcounts[i] = _m + 1;
			displs[i] = i * _m;
		}
		sendcounts[size-1] = (m - _m * (size-1)) + 1;
		displs[size-1] = (size-1) * _m;
			//IDL(INFO, printf("\ns1=%i, s2=%i, d1=%i, d2=%i\n",sendcounts[0],sendcounts[1],displs[0],displs[1]));
	}
	tempCrPointer = (int*)malloc((_m+1) * sizeof(int));
	if(rank!=0) {
		crs = (CRS*)malloc(sizeof(CRS));	// without MPI_Scatterv don't work
		crs->values = (DT*)malloc(sizeof(DT));
		crs->columnIndex = (int*)malloc(sizeof(int));
		crs->crPointer = (int*)malloc(sizeof(int));
	}
	MPI_Scatterv(crs->crPointer, sendcounts, displs, MPI_iDT, tempCrPointer, _m+1, MPI_iDT, 0, mycomm);
	if(rank==0) {
		free(sendcounts);
		free(displs);
	} else {
		first = tempCrPointer[0];
		for(i=0; i<_m+1; i++) tempCrPointer[i] -= first;
	}

	/* scatter the values/indices for each processor */
	if(rank==0) {
		sendcounts = (int*)malloc(size * sizeof(int));
		displs = (int*)malloc(size * sizeof(int));
		for(i=0; i<size-1; i++) {
			sendcounts[i] = (crs->crPointer)[i*_m + _m] - (crs->crPointer)[i*_m];
			displs[i] = (crs->crPointer)[i*_m];
		}
		sendcounts[size-1] = (crs->crPointer)[m] - (crs->crPointer)[(size-1)*_m];
		displs[size-1] = (crs->crPointer)[(size-1)*_m];
			//IDL(INFO, printf("\ns1=%i, s2=%i, d1=%i, d2=%i\n",sendcounts[0],sendcounts[1],displs[0],displs[1]));
	}
		//IDL(INFO, printf("\ntempCrPointer[_m]=%i\n",tempCrPointer[_m]));
	tempValues = (DT*)malloc(tempCrPointer[_m] * sizeof(DT));
	tempColumnIndex = (int*)malloc(tempCrPointer[_m] * sizeof(int));
	MPI_Scatterv(crs->values, sendcounts, displs, MPI_DT, tempValues, tempCrPointer[_m], MPI_DT, 0, mycomm);
	MPI_Scatterv(crs->columnIndex, sendcounts, displs, MPI_iDT, tempColumnIndex, tempCrPointer[_m], MPI_iDT, 0, mycomm);
	if(rank==0) {
		free(sendcounts);
		free(displs);
	}

	tempCRS->values = tempValues;
	tempCRS->sizeOfValues = tempCrPointer[_m];
	tempCRS->columnIndex = tempColumnIndex;
	tempCRS->sizeOfColumnIndex = tempCRS->sizeOfValues;
	tempCRS->crPointer = tempCrPointer;
	tempCRS->sizeOfcrPointer = _m + 1;

	//if(rank==0) MPI_clearCrsSparse(crs);

	IDL(INFO, printf("\n<----Exit MPI_scatterCRS() for rank=%i\n", rank));

	return tempCRS;
}

/*
* make a multiplication of the CRS part on the processor and the vector
*/
DT * MPI_CRSxVec(CRS * crs, int sizeOfB, DT * vec, int sizeOfVec) {
	int i, j, m, _m;
	int * recvcounts=NULL, * displs=NULL;
	DT * b;

	IDL(INFO, printf("\n---->Entered MPI_CRSxVec() for rank=%i\n", rank));

	m = sizeOfB;
	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	if(rank==0) {
		b = (DT*)calloc(m, sizeof(DT));
	} else {
		b = (DT*)calloc(_m, sizeof(DT));
	}

	MPI_Bcast(vec, sizeOfVec, MPI_DT, 0, mycomm);
	for(j=0; j<(*crs).sizeOfcrPointer-1; j++) {
		for(i=(*crs).crPointer[j]; i<(*crs).crPointer[j+1]; i++) {
			b[j] += (*crs).values[i] * vec[(*crs).columnIndex[i]];
		} 
	}
	
	if(rank==0) {
		recvcounts = (int*)malloc(size * sizeof(int));
		displs = (int*)malloc(size * sizeof(int));
		for(i=0; i<size-1; i++) recvcounts[i] = _m;
		recvcounts[size-1] = m - _m * (size-1);
		for(i=0; i<size; i++) displs[i] = i * _m;
	}
	MPI_Gatherv(b, _m, MPI_DT, b, recvcounts, displs, MPI_DT, 0, mycomm);

	IDL(INFO, printf("\n<----Exit MPI_CRSxVec() for rank=%i\n", rank));

	if(rank==0) {
		free(recvcounts);
		free(displs);
		return b;
	} else {
		free(b);
		return NULL;
	}
}

void MPI_printCrsSparse(CRS * crs) {
	int i;

//	if(rank==0) {
		IDL(INFO, printf("\n---->Entered MPI_printCrsSparse() for rank=%i\n", rank));
	
		printf("\nValues: ");
		for(i=0; i<(*crs).sizeOfValues; i++) {
			printf(FORMAT1,(*crs).values[i]);
		}
		printf("\nsizeOfValues: %i\n",(*crs).sizeOfValues);
	
		printf("\ncolumnIndex: ");
		for(i=0; i<(*crs).sizeOfColumnIndex; i++) {
			printf("%i, ",(*crs).columnIndex[i]);
		}
		printf("\nsizeOfColumnIndex: %i\n",(*crs).sizeOfColumnIndex);
	
		printf("\ncrPointer: ");
		for(i=0; i<(*crs).sizeOfcrPointer; i++) {
			printf("%i, ",(*crs).crPointer[i]);
		}
		printf("\nsizeOfcrPointer: %i\n",(*crs).sizeOfcrPointer);
	
		IDL(INFO, printf("<----Exit MPI_printCrsSparse() for rank=%i\n", rank));
//	}
}

/*
* deallocate the memory for the arrays of the CRS format
*/
void MPI_clearCrsSparse(CRS * crs) {
	IDL(INFO, printf("\n---->Entered MPI_clearCrsSparse() for rank=%i\n", rank));

	free((*crs).values);
	free((*crs).columnIndex);
	free((*crs).crPointer);
	free(crs);

	IDL(INFO, printf("\n<----Exit MPI_clearCrsSparse() for rank=%i\n", rank));
}


