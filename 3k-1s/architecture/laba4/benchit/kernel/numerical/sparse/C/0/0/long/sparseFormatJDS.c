/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatJDS.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/0/0/long/sparseFormatJDS.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/


#include "sparseFormatJDS.h"

/*
convert a normal matrix into the Jagged Diagonal Storage format
*/
JDS * convertToJdsSparse(DT ** matrix, long m, long n) {
	long i, j, maxNumberPerLine, temp, index;
	long c_values=0, c_col_ind=0; /* counter (c_values = c_col_ind) */
	SortedElem not_0_elem[m];
	JDS * jds;

	DT * jdValues;
	long * columnInd;
	long * perm;
	long * jdPtr;

	jds = (JDS*)malloc(sizeof(JDS));

	/* count all not-0-elements of the matrix per row */
	for(i=0; i<m; i++) {
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
	
	/* sort the number of not-0-elements in decreasing order */
	sort_QuickSort(not_0_elem,&not_0_elem[m-1]);
	
	/* create permutation */
	perm = (long*)malloc(m * sizeof(long));
	for(i=0; i<m; i++) {
		perm[i] = not_0_elem[i].oldPosition;
	}
	
	(*jds).permutation = perm;
	(*jds).sizeOfPermutation = m;

	/* create jd_pointer */
	maxNumberPerLine = not_0_elem[0].value;
	if(maxNumberPerLine == 0) {
		jdPtr = NULL;
	} else {
		jdPtr = (long*)malloc((maxNumberPerLine+1) * sizeof(long));	
		jdPtr[0] = 0;
		temp = 0;
		for(index=1; index<maxNumberPerLine; index++) {
			for(i=0; i<m; i++) {
				if(not_0_elem[i].value - (index-1) > 0) temp++;
			}
			jdPtr[index] = temp;
		}
		jdPtr[maxNumberPerLine] = c_values;
	}

	(*jds).jdPointer = jdPtr;
	(*jds).sizeOfjdPointer = maxNumberPerLine + 1;
	
	/* fill the array for values and column index */
	jdValues = (DT*)malloc(c_values * sizeof(DT));
	columnInd = (long*)malloc(c_values * sizeof(long));
	for(i=0; i<m; i++) {
		index = 0;
		for(j=0; j<n; j++) {
			if(matrix[perm[i]][j] != 0) {
				jdValues[jdPtr[index]+i] = matrix[perm[i]][j];
				columnInd[jdPtr[index]+i] = j;
				index++;
			}
		}
	}

	(*jds).values = jdValues;
	(*jds).sizeOfValues = c_values;

	(*jds).columnIndex = columnInd;
	(*jds).sizeOfColumnIndex = c_col_ind;

	return jds;
}

/*
multiply a JDS matrix with a full vector
*/
DT * JDSxVec(JDS * jds, DT * vec, long sizeOfVec) {
	long i, j, temp_start;
/*	DT * b_temp; */
	DT * b;

	if((*jds).sizeOfValues == 0) {
		return (DT*)calloc((*jds).sizeOfPermutation, sizeof(DT));
	}

/*	b_temp = (DT*)calloc((*jds).sizeOfPermutation, sizeof(DT)); */
	b = (DT*)calloc((*jds).sizeOfPermutation, sizeof(DT));
	
	/* Teil 1 f???r die ersten Spalten (ausgenommen letzte) der komprimierten Matrix */
	for(j=0; j<(*jds).sizeOfjdPointer-1; j++) {
		temp_start = (*jds).jdPointer[j];
		for(i=temp_start; i<(*jds).jdPointer[j+1]; i++) {
/*			b_temp[i-temp_start] += (*jds).values[i] * vec[(*jds).columnIndex[i]]; */
			b[(*jds).permutation[i-temp_start]] += (*jds).values[i] * vec[(*jds).columnIndex[i]];
		}
	}

/*	* Umordnung von b *
	b = (DT*)malloc((*jds).sizeOfPermutation * sizeof(DT));
	for(i=0; i<(*jds).sizeOfPermutation; i++) {
		b[(*jds).permutation[i]] = b_temp[i];
	}

	free(b_temp);
--> mit in Teil 1 integriert
*/

	return b;
}

void printJdsSparse(JDS * jds) {
	long i;

	printf("\nValues: ");
	for(i=0; i<(*jds).sizeOfValues; i++) {
		printf("%li, ",(*jds).values[i]);
	}
	printf("\nsizeOfValues: %li\n",(*jds).sizeOfValues);

	printf("\ncolumnIndex: ");
	for(i=0; i<(*jds).sizeOfColumnIndex; i++) {
		printf("%li, ",(*jds).columnIndex[i]);
	}
	printf("\nsizeOfColumnIndex: %li\n",(*jds).sizeOfColumnIndex);

	printf("\nPermutation: ");
	for(i=0; i<(*jds).sizeOfPermutation; i++) {
		printf("%li, ",(*jds).permutation[i]);
	}
	printf("\nsizeOfPermutation: %li\n",(*jds).sizeOfPermutation);

	printf("\njdPointer: ");
	for(i=0; i<(*jds).sizeOfjdPointer; i++) {
		printf("%li, ",(*jds).jdPointer[i]);
	}
	printf("\nsizeOfjdPointer: %li\n",(*jds).sizeOfjdPointer);

}

/*
deallocate the memory for the CRS matrix
*/
void clearJdsSparse(JDS * jds) {
	free((*jds).values);
	free((*jds).columnIndex);
	free((*jds).permutation);
	free((*jds).jdPointer);
	free(jds);
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


