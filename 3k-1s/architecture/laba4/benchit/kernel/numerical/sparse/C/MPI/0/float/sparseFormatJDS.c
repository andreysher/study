/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatJDS.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/float/sparseFormatJDS.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/
 
#include "sparseFormatJDS.h"
//#include "vector.h"

/*
* convert the part of the matrix on the processor into the JDS format
*/
JDS * MPI_convertToJdsSparse(DT ** matrix, int m, int n) {
	int i=0, j=0, _m=0, maxNumberPerLine=0, temp=0, index=0;
	int c_values=0, c_col_ind=0; /*Counter (c_values = c_col_ind)*/
	SortedElem * not_0_elem;
	JDS * jds;

	DT * jdValues;
	int * columnInd;
	int * perm;
	int * jdPtr;

	IDL(INFO, printf("\n---->Entered MPI_convertToJdsSparse() for rank=%i\n", rank));

	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	jds = (JDS*)malloc(sizeof(JDS));

	/*-count all not-0-elements of the matrixpart per row-----*/
	not_0_elem = (SortedElem*)malloc(_m * sizeof(SortedElem));
	for(i=0; i<_m; i++) {
		not_0_elem[i].value = 0;
		not_0_elem[i].oldPosition = i;
		for(j=0; j<n; j++) {
			if(matrix[i][j] != 0) {
				not_0_elem[i].value += 1;
				c_values++;
			}
		}
	}

	c_col_ind = c_values;

	IDL(INFO, printf("\nUnsorted not_0_elem: "));
	for(i=0; i<_m; i++) {
		IDL(INFO, printf("(%i, %i)",not_0_elem[i].value,not_0_elem[i].oldPosition));
	}
	/*--------------------------------------------------------*/

	/*-sort all not-0-elements of the matrixpart per row------*/
	sort_QuickSort(not_0_elem,&not_0_elem[_m-1]);
	IDL(INFO, printf("\nSorted not_0_elem: "));
	for(i=0; i<_m; i++) {
		IDL(INFO, printf("(%i, %i)",not_0_elem[i].value,not_0_elem[i].oldPosition));
	}
	/*--------------------------------------------------------*/
	
	/*-generate the permutation-------------------------------*/
	perm = (int*)malloc(_m * sizeof(int));
	for(i=0; i<_m; i++) {
		perm[i] = not_0_elem[i].oldPosition;
	}

	IDL(INFO, printf("\nPermutation: "));
	for(i=0; i<_m; i++) {
		IDL(INFO, printf("%i, ",perm[i]));
	}
	
	(*jds).permutation = perm;
	(*jds).sizeOfPermutation = _m;
	/*--------------------------------------------------------*/

	/*-generate the jdPointer---------------------------------*/
	maxNumberPerLine = not_0_elem[0].value;
	if(maxNumberPerLine == 0) {
		jdPtr = NULL;
	} else {
		jdPtr = (int*)malloc((maxNumberPerLine+1) * sizeof(int));	
		jdPtr[0] = 0;
		temp = 0;
		for(index=1; index<maxNumberPerLine; index++) {
			for(i=0; i<_m; i++) {
				if(not_0_elem[i].value - (index-1) > 0) temp++;
			}
			jdPtr[index] = temp;
		}
		jdPtr[maxNumberPerLine] = c_values;
	}

	IDL(INFO, printf("\njdPointer: "));
	for(i=0; i<maxNumberPerLine+1; i++) {
		IDL(INFO, printf("%i, ",jdPtr[i]));
	}

	(*jds).jdPointer = jdPtr;
	(*jds).sizeOfjdPointer = maxNumberPerLine + 1;
	free(not_0_elem);
	/*--------------------------------------------------------*/

	/*-detect the values/indices for the JDS format-----------*/
	jdValues = (DT*)malloc(c_values * sizeof(DT));
	columnInd = (int*)malloc(c_values * sizeof(int));
	for(i=0; i<_m; i++) {
		index = 0;
		for(j=0; j<n; j++) {
			if(matrix[perm[i]][j] != 0) {
				/* IDL(INFO, printf("\ni:%li, jdPtr[i]:%li, index:%li ",i,jdPtr[i],index)); */
				/* IDL(INFO, printf("\nperm[i]:%li, j:%li, matrix[perm[i]][j]:%d ",perm[i],j,matrix[perm[i]][j])); */
				jdValues[jdPtr[index]+i] = matrix[perm[i]][j];
				columnInd[jdPtr[index]+i] = j;
				index++;
			}
		}
	}

	IDL(INFO, printf("\njdValues: "));
	for(i=0; i<c_values; i++) {
		IDL(INFO, printf(FORMAT1,jdValues[i]));
	}
	IDL(INFO, printf("\ncol_ind: "));
	for(i=0; i<c_values; i++) {
		IDL(INFO, printf("%i, ",columnInd[i]));
	}

	(*jds).values = jdValues;
	(*jds).sizeOfValues = c_values;

	(*jds).columnIndex = columnInd;
	(*jds).sizeOfColumnIndex = c_col_ind;
	/*--------------------------------------------------------*/

	IDL(INFO, printf("\n<----Exit MPI_convertToJdsSparse() for rank=%i\n", rank));

	return jds;
}

/*
* root generate the complete JDS be get the parts of the processors
* in this operation the JDS part on the processors (inclusive root) will not be deallocated
*/
JDS * MPI_gatherJDS(JDS * jds, int m, int n) {
	int * node_sizeOfValues=NULL, * node_sizeOfjdPtr=NULL;
	int * recvcounts=NULL, * displs=NULL;
	int sizeOfTempValues=0, sizeOfTempJdPtr=0;
	int i=0, j=0, k=0, _rank=0, _m=0, start_val=0, start_jdPtr=0, col=0, perm_pos_in_gesamtJDS=0, sum_size=0;
	JDS * tempJDS=NULL;
	DT * tempValues=NULL;
	int * tempColumnIndex=NULL, * tempPerm=NULL, * tempJdPointer=NULL;
	DT * recvbuf_values=NULL;
	int * recvbuf_colIndex=NULL, * node_jdPtr=NULL, * node_perm=NULL;
	SortedElem * tempMaxPerLine=NULL;

	IDL(INFO, printf("\n---->Entered MPI_gatherJDS() for rank=%i\n", rank));
	
	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	tempJDS = (JDS*)malloc(sizeof(JDS));

	/* get the number of values/indices of the JDS format on each processor */
	if(rank==0) node_sizeOfValues = (int*)malloc(size * sizeof(int));
	MPI_Gather(&(jds->sizeOfValues), 1, MPI_iDT, node_sizeOfValues, 1, MPI_iDT, 0, mycomm);
	/* sum this values */
	if(rank==0) {
		for(i=0; i<size; i++) sizeOfTempValues += node_sizeOfValues[i];
	}
	IDL(INFO, printf("\nsizeOfTempValues = %i",sizeOfTempValues));
	
	/* get the number of jdPointer of the JDS format on each processor */
	if(rank==0) node_sizeOfjdPtr = (int*)malloc(size * sizeof(int));
	MPI_Gather(&((*jds).sizeOfjdPointer), 1, MPI_iDT, node_sizeOfjdPtr, 1, MPI_iDT, 0, mycomm);
	//if(rank==0) IDL(INFO,MPI_printVector(node_sizeOfjdPtr,size));

	/* recieve the jdPointer of each processor */
	if(rank==0) {
		for(i=0; i<size; i++) sum_size += node_sizeOfjdPtr[i];
		displs = (int*)calloc(size, sizeof(int));
		for(i=1; i<size; i++) displs[i] = displs[i-1] + node_sizeOfjdPtr[i-1];
		node_jdPtr = (int*)malloc(sum_size * sizeof(int));
	}
	MPI_Gatherv(jds->jdPointer, jds->sizeOfjdPointer, MPI_iDT, node_jdPtr, node_sizeOfjdPtr, displs, MPI_iDT, 0, mycomm);
	//if(rank==0) IDL(INFO,MPI_printVector(node_jdPtr,sum_size));
	if(rank==0) free(displs);

	/* recieve the permutation of each processor */
	if(rank==0) {
		recvcounts = (int*)malloc(size * sizeof(int));
		displs = (int*)malloc(size * sizeof(int));
		for(i=0; i<size; i++) {
			recvcounts[i] = _m;
			displs[i] = i * _m;
		}
		recvcounts[size-1] = m - _m * (size-1);
		node_perm = (int*)malloc(m * sizeof(int));
	}
	MPI_Gatherv(jds->permutation, jds->sizeOfPermutation, MPI_iDT, node_perm, recvcounts, displs, MPI_iDT, 0, mycomm);
	//if(rank==0) IDL(INFO,MPI_printVector(node_perm,m));
	if(rank==0) {
		free(recvcounts);
		free(displs);
	}

	/* generate for each row of the complete JDS the number of elements (not equals 0), save in tempMaxPerLine */
	if(rank==0) {
		tempMaxPerLine = (SortedElem*)malloc(m * sizeof(SortedElem));
		for(i=0; i<m; i++) {
			tempMaxPerLine[i].value = 0;
			tempMaxPerLine[i].oldPosition = i;
		}
		start_jdPtr = 0;
		for(i=0; i<size-1; i++) {
			for(j=0; j<node_sizeOfjdPtr[i]-1; j++) {
				for(k=0; k<node_jdPtr[start_jdPtr+j+1]-node_jdPtr[start_jdPtr+j]; k++) {
					tempMaxPerLine[ i*_m + node_perm[i*_m+k] ].value++;
				}
			}
			start_jdPtr += node_sizeOfjdPtr[i];
		}
		for(j=0; j<node_sizeOfjdPtr[size-1]-1; j++) {
			for(k=0; k<node_jdPtr[start_jdPtr+j+1]-node_jdPtr[start_jdPtr+j]; k++) {
				tempMaxPerLine[ (size-1)*_m + node_perm[(size-1)*_m+k] ].value++;
			}
		}
	}

	/* sort the array tempMaxPerLine */
	if(rank==0) {
			// IDL(INFO,printf("\n{%i, %i, %i, %i, %i, %i, %i}",tempMaxPerLine[0].value, tempMaxPerLine[1].value, tempMaxPerLine[2].value,tempMaxPerLine[3].value, tempMaxPerLine[4].value, tempMaxPerLine[5].value, tempMaxPerLine[6].value));
			// IDL(INFO,printf("\n{%i, %i, %i, %i, %i, %i, %i}",tempMaxPerLine[0].oldPosition, tempMaxPerLine[1].oldPosition, tempMaxPerLine[2].oldPosition,tempMaxPerLine[3].oldPosition, tempMaxPerLine[4].oldPosition, tempMaxPerLine[5].oldPosition, tempMaxPerLine[6].oldPosition));
		sort_QuickSort(tempMaxPerLine, &tempMaxPerLine[m-1]);
			// IDL(INFO,printf("\n{%i, %i, %i, %i, %i, %i, %i}",tempMaxPerLine[0].value, tempMaxPerLine[1].value, tempMaxPerLine[2].value,tempMaxPerLine[3].value, tempMaxPerLine[4].value, tempMaxPerLine[5].value, tempMaxPerLine[6].value));
			// IDL(INFO,printf("\n{%i, %i, %i, %i, %i, %i, %i}",tempMaxPerLine[0].oldPosition, tempMaxPerLine[1].oldPosition, tempMaxPerLine[2].oldPosition,tempMaxPerLine[3].oldPosition, tempMaxPerLine[4].oldPosition, tempMaxPerLine[5].oldPosition, tempMaxPerLine[6].oldPosition));

		/* generate the permutation out of the sorted list */
		tempPerm = (int*)malloc(m * sizeof(int));
		for(i=0; i<m; i++) tempPerm[i] = tempMaxPerLine[i].oldPosition;

		/* generate the jdPointer for the complete JDS */
		tempJdPointer = (int*)malloc((tempMaxPerLine[0].value+1) * sizeof(int));
		tempJdPointer[0] = 0;
		sizeOfTempJdPtr = tempMaxPerLine[0].value + 1;
		for(i=1; i<sizeOfTempJdPtr; i++) {
				// IDL(INFO,printf("\n{%i, %i, %i, %i, %i, %i}",tempMaxPerLine[0].value, tempMaxPerLine[1].value, tempMaxPerLine[2].value,tempMaxPerLine[3].value, tempMaxPerLine[4].value, tempMaxPerLine[5].value));
			tempJdPointer[i] = tempJdPointer[i-1];
			for(j=0; j<m; j++) {
				if(tempMaxPerLine[j].value > 0) {
					tempJdPointer[i]++;
					tempMaxPerLine[j].value--;
				} else break;
			}
		}
	}

	/* recieve the values/indices from all processors */
	if(rank==0) {
		recvbuf_values = (DT*)malloc(sizeOfTempValues * sizeof(DT));
		tempValues = (DT*)malloc(sizeOfTempValues * sizeof(DT));
		recvbuf_colIndex = (int*)malloc(sizeOfTempValues * sizeof(int));
		tempColumnIndex = (int*)malloc(sizeOfTempValues * sizeof(int));
		displs = (int*)malloc(size * sizeof(int));
		displs[0] = 0;
		for(i=1; i<size; i++) {
			displs[i] = displs[i-1] + node_sizeOfValues[i-1];
		}
	}
	MPI_Gatherv(jds->values, jds->sizeOfValues, MPI_DT, recvbuf_values, node_sizeOfValues, displs, MPI_iDT, 0, mycomm);
	//if(rank==0) IDL(INFO, MPI_printVector(recvbuf_values,sizeOfTempValues));
	MPI_Gatherv(jds->columnIndex, jds->sizeOfColumnIndex, MPI_iDT, recvbuf_colIndex, node_sizeOfValues, displs, MPI_iDT, 0, mycomm);
	free(displs);

	/* generate the arrays of the values and indices for the complete JDS */
	if(rank==0) {
		start_val = 0;
		start_jdPtr = 0;
		for(_rank=0; _rank<size; _rank++) {				// run for each node
			for(i=0; i<node_sizeOfjdPtr[_rank]-1; i++) {		// run thru the jdPtr of each node
				for(j=node_jdPtr[start_jdPtr+i]; j<node_jdPtr[start_jdPtr+i+1]; j++) {
					//value = recvbuf_values[start_val+j];
					//colIndex = recvbuf_colIndex[start_val+j];
					col = _rank*_m + node_perm[(_rank*_m)+j-node_jdPtr[start_jdPtr+i]];
					for(k=0; k<m; k++) {
						if(col==tempMaxPerLine[k].oldPosition) {
							perm_pos_in_gesamtJDS = k;
							break;
						}
					}
					//tempValues[tempJdPointer[i]+perm_pos_in_gesamtJDS] = value;
					//tempColumnIndex[tempJdPointer[i]+perm_pos_in_gesamtJDS] = colIndex;
					tempValues[tempJdPointer[i]+perm_pos_in_gesamtJDS] = recvbuf_values[start_val+j];
					tempColumnIndex[tempJdPointer[i]+perm_pos_in_gesamtJDS] = recvbuf_colIndex[start_val+j];
				}
			}
			start_jdPtr += node_sizeOfjdPtr[_rank];
			start_val += node_sizeOfValues[_rank];
		}
	}

	if(rank==0) {
		free(node_sizeOfValues);
		free(node_sizeOfjdPtr);
		free(node_jdPtr);
		free(node_perm);
		free(tempMaxPerLine);
		free(recvbuf_values);
		free(recvbuf_colIndex);
	}

	IDL(INFO, printf("\n<----Exit MPI_gatherJDS() for rank=%i\n", rank));

	if(rank==0) {
		tempJDS->values = tempValues;
		tempJDS->sizeOfValues = sizeOfTempValues;
		tempJDS->columnIndex = tempColumnIndex;
		tempJDS->sizeOfColumnIndex = sizeOfTempValues;
		tempJDS->permutation = tempPerm;
		tempJDS->sizeOfPermutation = m;
		tempJDS->jdPointer = tempJdPointer;
		tempJDS->sizeOfjdPointer = sizeOfTempJdPtr;
	} else {
		/* return for all processors excluding root an "empty" JDS */
		tempJDS->values = (DT*)malloc(sizeof(DT));
		tempJDS->sizeOfValues = -1;
		tempJDS->columnIndex = (int*)malloc(sizeof(int));
		tempJDS->sizeOfColumnIndex = -1;
		tempJDS->permutation = (int*)malloc(sizeof(int));
		tempJDS->sizeOfPermutation = -1;
		tempJDS->jdPointer = (int*)malloc(sizeof(int));
		tempJDS->sizeOfjdPointer = -1;
	}

	return tempJDS;
}

#ifdef hahah
/*
* root generate out of the complete JDS the parts for all processors
* in this operation the complete JDS part on root will be not deallocated
* (inverse operation of MPI_gatherCRS)
*/
JDS * MPI_scatterJDS(JDS * jds, int m, int n) {

	IDL(INFO, printf("\n---->Entered MPI_scatterJDS() for rank=%i\n", rank));

	tempJDS->values = tempValues;
	tempJDS->sizeOfValues = ;
	tempJDS->columnIndex = tempColumnIndex;
	tempJDS->sizeOfColumnIndex = tempJDS->sizeOfValues;
	tempJDS->permutation = tempPerm;
	tempJDS->sizeOfPermutation = ;
	tempJDS->jdPointer = tempJdPointer;
	tempJDS->sizeOfjdPointer = ;

	if(rank==0) MPI_clearCrsSparse(jds);

	IDL(INFO, printf("\n<----Exit MPI_scatterJDS() for rank=%i\n", rank));

	return tempJDS;
}
#endif

/*
* make a multiplication of the JDS part on the processor and the vector
*/
DT * MPI_JDSxVec(JDS * jds, int sizeOfB, DT * vec, int sizeOfVec) {
	int i=0, j=0, _m=0, temp_start=0;
	DT * b=NULL;
	int * recvcounts=NULL, * displs=NULL;

	IDL(INFO, printf("\n---->Entered MPI_JDSxVec() for rank=%i\n", rank));

	_m = ceil(sizeOfB / (float)size);
	if(rank+1 == size) {
		_m = sizeOfB - _m * (size-1);
	}

	if(rank==0) {
		b = (DT*)calloc(sizeOfB, sizeof(DT));
	} else {
		b = (DT*)calloc((*jds).sizeOfPermutation, sizeof(DT));
	}

	MPI_Bcast(vec, sizeOfVec, MPI_DT, 0, mycomm);
	for(j=0; j<(*jds).sizeOfjdPointer-1; j++) {
		temp_start = (*jds).jdPointer[j];
		for(i=temp_start; i<(*jds).jdPointer[j+1]; i++) {
			b[(*jds).permutation[i-temp_start]] += (*jds).values[i] * vec[(*jds).columnIndex[i]];
		}
	}

	if(rank==0) {
		recvcounts = (int*)malloc(size * sizeof(int));
		displs = (int*)malloc(size * sizeof(int));
		for(i=0; i<size-1; i++) recvcounts[i] = _m;
		recvcounts[size-1] = sizeOfB - _m * (size-1);
		for(i=0; i<size; i++) displs[i] = i * _m;
	}
	MPI_Gatherv(b, _m, MPI_DT, b, recvcounts, displs, MPI_DT, 0, mycomm);
	
	IDL(INFO, printf("\n<----Exit MPI_JDSxVec() for rank=%i\n", rank));

	if(rank==0) {
		free(recvcounts);
		free(displs);
		return b;
	} else {
		free(b);
		return NULL;
	}
}

void MPI_printJdsSparse(JDS * jds) {
	int i;

	IDL(INFO, printf("\n---->Entered MPI_printJdsSparse() for rank=%i\n", rank));

//	if(rank==0) {
		printf("\nValues: ");
		for(i=0; i<(*jds).sizeOfValues; i++) {
			printf(FORMAT1,(*jds).values[i]);
		}
		printf("\nsizeOfValues: %i\n",(*jds).sizeOfValues);
	
		printf("\ncolumnIndex: ");
		for(i=0; i<(*jds).sizeOfColumnIndex; i++) {
			printf("%i, ",(*jds).columnIndex[i]);
		}
		printf("\nsizeOfColumnIndex: %i\n",(*jds).sizeOfColumnIndex);
	
		printf("\nPermutation: ");
		for(i=0; i<(*jds).sizeOfPermutation; i++) {
			printf("%i, ",(*jds).permutation[i]);
		}
		printf("\nsizeOfPermutation: %i\n",(*jds).sizeOfPermutation);
	
		printf("\njdPointer: ");
		for(i=0; i<(*jds).sizeOfjdPointer; i++) {
			printf("%i, ",(*jds).jdPointer[i]);
		}
		printf("\nsizeOfjdPointer: %i\n",(*jds).sizeOfjdPointer);
//	}

	IDL(INFO, printf("<----Exit MPI_printJdsSparse() for rank=%i\n", rank));
}

/*
* deallocate the memory for the arrays of the JDS format
*/
void MPI_clearJdsSparse(JDS * jds) {
	IDL(INFO, printf("\n---->Entered MPI_clearJdsSparse() for rank=%i\n", rank));

	free((*jds).values);
	free((*jds).columnIndex);
	free((*jds).permutation);
	free((*jds).jdPointer);
	free(jds);

	IDL(INFO, printf("\n<----Exit MPI_clearJdsSparse() for rank=%i\n", rank));
}

void sort_QuickSort(SortedElem *links, SortedElem *rechts) {
	SortedElem *ptr1 = links;
	SortedElem *ptr2 = rechts;
	SortedElem w, x;

	x = *(links + ((rechts - links) >> 1));
	do {
		while((*ptr1).value > x.value) {
			ptr1++;
		}
		while((*ptr2).value < x.value) {
			ptr2--;
		}
		if(ptr1 > ptr2) break;
		w = *ptr1;
		*ptr1 = *ptr2;
		*ptr2 = w;
	} while(++ptr1 <= --ptr2);

	if(links < ptr2) {
		sort_QuickSort(links, ptr2);
	}

	if(ptr1 < rechts) {
		sort_QuickSort(ptr1, rechts);
	}
}


