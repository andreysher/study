/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matrix.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/0/0/double/matrix.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "matrix.h"

/*
create an empty matrix (only the allocated memory)
*/
DT ** createMatrix(long m, long n) {
	long i;
	DT ** matrix;

	matrix = (DT**)malloc(m * sizeof(DT*));
	assert(matrix != NULL);
	matrix[0] = (DT*)malloc(m * n * sizeof(DT));
	assert(matrix[0] != NULL);
	for(i=1; i<m; i++) matrix[i] = matrix[0] + i*n;

	return matrix;
}

/*
deallocate the memory for the matrix
*/
void clearMatrix(DT ** matrix) {
	free(matrix[0]);
	free(matrix);
}

/*
fill the matrix with random numbers
	x in [1, 2, ... , 9] for long
	x in [1.0, ..., 9.0] for double
for percent > 30.0 the algorithm try to fill the complete matrix
for percent <=30.0 the algorithm garantated a better filling for "small" matrices or small percentage
by generating the index for the random values
*/
void initRandomMatrix(DT ** matrix, long m, long n, double percent) {
	long i, j, k, count;
	int r;

	initZERO(matrix, m, n);

	if(percent > 30.0) {
		for(i=0; i<m; i++) {
			for(j=0; j<n; j++) {
				r = rand();
				if((double)r*(100.0/(double)RAND_MAX) < percent) {
					matrix[i][j] = 1 + (DT)(9.0*rand()/(RAND_MAX+1.0));
				} else {
					matrix[i][j] = 0;
				}
			}
		}
	} else {
		count = ceil(m*n * percent/100.0);
		for(k=0; k<count; k++) {
			i = (long)((double)m*rand()/(RAND_MAX+1.0));
			j = (long)((double)n*rand()/(RAND_MAX+1.0));
			matrix[i][j] = 1 + (DT)(9.0*rand()/(RAND_MAX+1.0));
		}
	}

}

/*
fill the matrix with 0
*/
void initZERO(DT ** matrix, long m, long n) {
	long i, j;

	for(i=0; i<m; i++) {
		for(j=0; j<n; j++) {
			matrix[i][j] = 0;
		}
	}

}

/*
create an identity matrix
*/
void initIDENTITY(DT ** matrix, long m, long n) {
	long i, index;

	initZERO(matrix, m, n);

	index = n;
	if(m<n) index = m;

	for(i=0; i<index; i++) {
		matrix[i][i] = 1;
	}

}

/*
create a diagonal matrix with random numbers
	x in [1, 2, ... , 9] for long
	x in [1.0, ..., 9.0] for double
diag = 0 -> 1 diagonal
diag = 1 -> 3 diagonals
...
*/
void initDIAG(DT ** matrix, long m, long n, long diag) {
	long i, j, index, dInd;

	initZERO(matrix, m, n);

	index = n;
	if(m < n) index = m;

	if(diag < 0) diag = 0;
	if(diag > index-1) diag = index-1;

	dInd = diag;
	for(i=0; i<index; i++) {
		for(j=-dInd; j<=dInd; j++) {
			if(0<=i+j && i+j<n) matrix[i][i+j] = 1 + (DT)(9.0*rand()/(RAND_MAX+1.0));
		}
	}

}

/*
help-function for init5PSTAR() to set one row of the matrix 0 and insert one 1
*/
void resetLine(DT ** matrix, long n, long index) {
	long i;
	for(i=0; i<n; i++) {
		matrix[index][i] = 0;
	}
	matrix[index][index] = 1;
}

/*
create a 5-point-star matrix for finit elements 
*/
void init5PSTAR(DT ** matrix, long m, long n) {
	long i, index, a;

	initZERO(matrix, m, n);

	index = n;
	if(m < n) index = m;

	if(index<9) {
		for(i=0; i<index; i++) {
			matrix[i][i] = 4;
		}
		return;
	}

	/* create main diagonal */
	matrix[0][0] = 4;
	matrix[0][1] = -1;
	for(i=1; i<index-1; i++) {
		matrix[i][i-1] = -1;
		matrix[i][i] = 4;
		matrix[i][i+1] = -1;
	}
	matrix[index-1][index-1] = 4;
	matrix[index-1][index-2] = -1;

	/* create secondary diagonal */
	a = floor(sqrt(index));
	for(i=0; i<index-a; i++) {
		matrix[i][i+a] = -1;
	}
	for(i=a; i<index; i++) {
		matrix[i][i-a] = -1;
	}
	
	/* set boundary point */
	for(i=0; i<a; i++) {
		resetLine(matrix, n, i);
	}
	for(i=index-a; i<index; i++) {
		resetLine(matrix, n, i);
	}
	for(i=a; i<index-a; i=i+a) {
		resetLine(matrix, n, i);
		resetLine(matrix, n, i+a-1);
	}
}

/*
make a multiplication of an matrix and vector
*/
DT * MatxVec(DT ** matrix, long m, long n, DT * vec, long sizeOfVec) {
	long i, j;
	DT * b;

	if(n != sizeOfVec) {
		printf("\n\n--Error-- incompartible size in MatxVec()");
		exit(1);
	}
	
	b = (DT*)calloc(m, sizeof(DT));
	for(i=0; i<m; i++) {
		for(j=0; j<n; j++) {
			/* b[i] += matrix[i][j] * vec[j]; */
			b[i] += matrix[0][i*n+j] * vec[j];
		}
	}

	return b;
}

void printMatrix(DT ** matrix, long m, long n) {
	long i, j;

	for(i=0; i<m; i++) {
		for(j=0; j<n-1; j++) {
			printf("%e, ",matrix[i][j]);
		}
		printf("%e\n",matrix[i][n-1]);
	}
}


