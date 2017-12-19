/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matrix.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/double/matrix.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "matrix.h"

/*
* deliver max{ i1, i2 }
*/
int iMax(int i1, int i2) {
	if(i1 > i2) {
		return i1;
	} else {
		return i2;
	}
}

/*
* deliver min{ i1, i2 }
*/
int iMin(int i1, int i2) {
	if(i1 < i2) {
		return i1;
	} else {
		return i2;
	}
}

/*
* create an empty part of matrix for each of the n processors
* processor 1 to n-1 get the same number of rows of the matrix
* the last processor get a number of rows less than for the others 
*/
DT ** MPI_createMatrix(int m, int n) {
	int i, _m;
	DT ** matrix;

	IDL(INFO, printf("\n---->Entered MPI_createMatrix() for rank=%i\n", rank));

	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	matrix = (DT**)malloc(_m * sizeof(DT*));
	assert(matrix != NULL);
	matrix[0] = (DT*)malloc(_m * n * sizeof(DT));
	assert(matrix[0] != NULL);
	for(i=1; i<_m; i++) matrix[i] = matrix[0] + i*n;

	IDL(INFO, printf("\n<----Exit MPI_createMatrix() for rank=%i\n", rank));

	return matrix;
}

/*
* deallocate the memory for the matrix
*/
void MPI_clearMatrix(DT ** matrix) {
	IDL(INFO, printf("\n---->Entered MPI_clearMatrix() for rank=%i\n", rank));

	free(matrix[0]);
	free(matrix);

	IDL(INFO, printf("\n<----Exit MPI_clearMatrix() for rank=%i\n", rank));
}

/*
* fill the matrix with random numbers
* 	x in [1, 2, ... , 9] for long
* 	x in [1.0, ..., 9.0] for double
* for percent > 30.0 the algorithm try to fill the complete matrix
* for percent <=30.0 the algorithm garantated a better filling for "small" matrices or small percentage
* by generating the index for the random values
*/
void MPI_initRandomMatrix(DT ** matrix, int m, int n, float percent) {
	int i, j, k, count, _m;
	int r;

	IDL(INFO, printf("\n---->Entered MPI_initRandomMatrix() for rank=%i\n", rank));

	MPI_initZERO(matrix, m, n);

	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	if(percent > 30.0) {
		IDL(INFO, printf("\nuse version 1 for MPI_initRandom"));
		for(i=0; i<_m; i++) {
			for(j=0; j<n; j++) {
					//IDL(INFO, printf("\n(i,j) = (%li, %li)\n",i,j));
				r = rand();
				if((float)r*(100.0/(float)RAND_MAX) < percent) {
					matrix[i][j] = 1 + (DT)(9.0*rand()/(RAND_MAX+1.0));
				}
			}
		}
	} else {
		IDL(INFO, printf("\nuse version 2 für MPI_initRandom"));
		/* dieser Algo hat eine bessere Auslastung der Matrix zufolge */
		count = ceil(_m*n * percent/100.0);
		for(k=0; k<count; k++) {
			i = (int)((float)_m*rand()/(RAND_MAX+1.0));
			j = (int)((float)n*rand()/(RAND_MAX+1.0));
				//IDL(INFO, printf("\n(i,j) = (%li, %li)\n",i,j));
			matrix[i][j] = 1 + (DT)(9.0*rand()/(RAND_MAX+1.0));
		}
	}

	IDL(INFO, printf("\n<----Exit MPI_initRandomMatrix() for rank=%i\n", rank));
}

/*
* fill the matrix with 0
*/
void MPI_initZERO(DT ** matrix, int m, int n) {
	int i, j, index;

	IDL(INFO, printf("\n---->Entered MPI_initZERO() for rank=%i\n", rank));

	index = ceil(m / (float)size);
	if(rank+1 == size) {
		index = m - index * (size-1);
	}

	for(i=0; i<index; i++) {
		for(j=0; j<n; j++) {
			matrix[i][j] = 0;
		}
	}

	IDL(INFO, printf("\n<----Exit MPI_initZERO() for rank=%i\n", rank));
}

/*
* create an identity matrix
*/
void MPI_initIDENTITY(DT ** matrix, int m, int n) {
	int i, index, _m, temp;

	IDL(INFO, printf("\n---->Entered initIDENTITY() for rank=%i\n", rank));

	MPI_initZERO(matrix, m, n);

	_m = ceil(m / (float)size);
	temp = _m;
	if(rank+1==size) {
		_m = m - _m * (size-1);
	}

	index = iMin(m, n);

	for(i=0; i<_m; i++) {
		if(rank*temp+i<index) {
			matrix[i][rank*temp+i] = 1;
		}
	}

	IDL(INFO, printf("\n<----Exit initIDENTITY() for rank=%i\n", rank));
}

/*
* create a diagonal matrix with random numbers
* 	x in [1, 2, ... , 9] for long
* 	x in [1.0, ..., 9.0] for double
* diag = 0 -> 1 diagonal
* diag = 1 -> 3 diagonals
* ...
*/
void MPI_initDIAG(DT ** matrix, int m, int n, int diag) {
	int i, j, index, _index, pos;

	IDL(INFO, printf("\n---->Entered MPI_initDIAG() for rank=%i\n", rank));

	MPI_initZERO(matrix, m, n);

	index = iMin(m, n);

	_index = ceil(m / (float)size);
	pos = rank * _index;

	if(rank+1 == size) {
		_index = m - _index * (size-1);
	}

	if(diag < 0) diag = 0;
	if(diag > index-1) diag = index-1;

		//MPI_Barrier(mycomm);
	/* erzeuge oberes Trapez (z.B. diag=2):
	* -> * * *
	*    * * * *
	*    * * * * *
	*/
	if(pos<diag) {
		for(i=pos; i<iMin(pos+_index,diag); i++) {
			for(j=0; j<=i+diag; j++) {
					//printf("\np1: rank%i:  (i,j) = (%li, %li)",rank,i-pos,j);fflush(stdout);fflush(stderr);
				matrix[i-pos][j] = 1 + (DT)(9.0*rand()/(RAND_MAX+1.0));
			}
		}
	}

		//MPI_Barrier(mycomm);
	/* erzeuge mittleres Parallelogram (z.B. diag=2):
	* ->   * * * * *
	*        * * * * *
	*          * * * * *
	*/
	if(diag<=pos || diag<=pos+_index) {
		for(i=iMax(pos,diag); i<iMin(pos+_index,index-diag); i++) {
			for(j=-diag; j<=diag; j++) {
					//printf("\np2: rank%i:  (i,j) = (%li, %li)",rank,i-pos,i+j);fflush(stdout);fflush(stderr);
				matrix[i-pos][i+j] = 1 + (DT)(9.0*rand()/(RAND_MAX+1.0));
			}
		}
	}

		//MPI_Barrier(mycomm);
	/* erzeuge unteres Trapez (z.B. diag=2):
	* ->         * * * * *
	*              * * * * 
	*                * * * 
	*/
	if(index-diag<=pos || index-diag<=pos+_index) {
		for(i=iMax(pos,diag); i<iMin(pos+_index,index); i++) {
			for(j=i-diag; j<iMin(i+diag,index); j++) {
					//printf("\np3: rank%i:  (i,j) = (%li, %li)",rank,i-pos,j);fflush(stdout);fflush(stderr);
				matrix[i-pos][j] = 1 + (DT)(9.0*rand()/(RAND_MAX+1.0));
			}
		}
	}

	IDL(INFO, printf("\n<----Exit MPI_initDIAG() for rank=%i\n", rank));
}

/* wird in init5PSTAR benutzt */
void MPI_resetLine(DT ** matrix, int n, int ind1, int ind2) {
	int i;

		//printf("\nresline: rank%i:  (n, ind1, ind2) = (%li, %li, %li)",rank, n, ind1, ind2);fflush(stdout);fflush(stderr);
	for(i=0; i<n; i++) {
		matrix[ind1][i] = 0;
	}
	matrix[ind1][ind2] = 1;
}

void MPI_init5PSTAR(DT ** matrix, int m, int n) {
	int i, index, _index, a, pos;

	IDL(INFO, printf("\n---->Entered MPI_init5PStar() for rank=%i\n", rank));

	MPI_initZERO(matrix, m, n);

	index = iMin(m, n);

	/*
	* evaluate the size of the biggest possible square 5-point-star, which fit in the matrix
	*/
	index = floor(sqrt((float)index));
	index = index * index;

	_index = ceil(m / (float)size);			/* number of rows for the matrixpart of the processor */
	pos = rank * _index;				/* absolute position in the matrix of row 0 (from the matrix on this processor) */

	if(rank + 1 == size) {
		_index = m - _index * (size-1);		/* number of rows for the last processor */
	}

	if(index<9) {
		for(i=iMax(pos,0); i<iMin(pos+_index,index); i++) {
			matrix[i-pos][i] = 1;
		}
		return;
	}

		//MPI_Barrier(mycomm);
		//printf("\n1: rank%i:  (index, _index, pos) = (%li, %li, %li)",rank, index, _index, pos);fflush(stdout);fflush(stderr);
	/* create main diagonal */
	if(rank==0) { matrix[0][0] = 4; matrix[0][0] = -1; }
	for(i=iMax(pos,1); i<iMin(pos+_index,index-1); i++) {
		matrix[i-pos][i-1] = -1;
		matrix[i-pos][i] = 4;
		matrix[i-pos][i+1] = -1;
	}
	if(pos+_index == index) {
		matrix[index-pos-1][index-2] = -1;
		matrix[index-pos-1][index-1] = 4;
	}

		//MPI_Barrier(mycomm);
		//printf("\n2: rank%i:  (index, _index, pos) = (%li, %li, %li)",rank, index, _index, pos);fflush(stdout);fflush(stderr);
	/* create secondary diagonal */
	a = floor(sqrt((float)index));
	for(i=iMax(pos,0); i<iMin(pos+_index,index-a); i++) {
		matrix[i-pos][i+a] = -1;
	}
	for(i=iMax(pos,a); i<iMin(pos+_index,index); i++) {
		matrix[i-pos][i-a] = -1;
	}
	
		//MPI_Barrier(mycomm);
		//printf("\n3: rank%i:  (index, _index, pos, a) = (%li, %li, %li, %li)",rank, index, _index, pos, a);fflush(stdout);fflush(stderr);
	/* set boundary point */
	for(i=iMax(pos,0); i<iMin(pos+_index,a); i++) {
		MPI_resetLine(matrix, n, i - pos, i);				/* boundary point at the bottom */
	}
	for(i=iMax(pos,index-a); i<iMin(pos+_index,index); i++) {
		MPI_resetLine(matrix, n, i - pos, i);				/* boundary point at the top */
	}
	for(i=a; i<index-a; i=i+a) {
		if(i-pos>=0 && i-pos<_index) MPI_resetLine(matrix, n, i - pos, i);	/* boundary point left */
	}
	for(i=2*a-1; i<index; i=i+a) {
		if(i-pos>=0 && i-pos<_index) MPI_resetLine(matrix, n, i - pos, i);	/* boundary point right */
	}

	IDL(INFO, printf("\n<----Exit MPI_init5PStar() for rank=%i\n", rank));
}

/*
* make a multiplication of the matrixpart on the processor and the vector
*/
DT * MPI_MatxVec(DT ** matrix, int m, int n, DT * x, int sizeOfX) {
	int i=0, j=0, _m=0;
	int * recvcounts=NULL, * displs=NULL;
	DT * b=NULL;

	IDL(INFO, printf("\n---->Entered MPI_MatxVec() for rank=%i\n", rank));

	if(n != sizeOfX) {
		printf("\n\n--Error-- inkompartible Groessen in MatxVec()");
		exit(1);
	}

	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	if(rank==0) {
		b = MPI_createVector(m);
	} else {
		b = MPI_createVector(_m);
	}	

	/* root shares the vector with the others */
	MPI_Bcast(x, sizeOfX, MPI_DT, 0, mycomm);
		//IDL(INFO, MPI_printVector(x, sizeOfX));
	
	for(i=0; i<_m; i++) {
		for(j=0; j<n; j++) {
			/* b_temp[i] += matrix[i][j] * x[j]; */
			b[i] += matrix[0][i*n+j] * x[j];
		}
	}
		//IDL(INFO, MPI_printVector(b_temp, _m));
	
	if(rank==0) {
		recvcounts = (int*)malloc(size * sizeof(int));
		displs = (int*)malloc(size * sizeof(int));
		for(i=0; i<size-1; i++) recvcounts[i] = _m;
		recvcounts[size-1] = m - _m * (size-1);
		for(i=0; i<size; i++) displs[i] = i * _m;
	}

	/* root combine the result of each processor */
	MPI_Gatherv(b, _m, MPI_DT, b, recvcounts, displs, MPI_DT, 0, mycomm);
		//if(rank==0) IDL(INFO, MPI_printVector(b, m));

	IDL(INFO, printf("\n<----Exit MPI_MatxVec() for rank=%i\n", rank));

	if(rank==0){
		free(displs);
		free(recvcounts);
		return b;
	} else {
		MPI_clearVector(b);
		return NULL;
	}
}

/*
* root combine the matrixparts to one matrix and make the multiplicator only on root, it's to verify the multiplication
*/
DT * MPI_MatxVec_pruef(DT ** matrix, int m, int n, DT * x, int sizeOfX) {
	int i, j, _m;
	int * recvcounts=NULL, * displs=NULL;
	DT * b=NULL;
	DT ** tempMatrix;
	
	IDL(INFO, printf("\n---->Entered MPI_MatxVec_pruef() for rank=%i\n", rank));

	if(n != sizeOfX) {
		printf("\n\n--Error-- inkompartible Groessen in MatxVec()");
		exit(1);
	}
	
	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	if(rank==0) {
		b = MPI_createVector(m);
		tempMatrix = MPI_createMatrix(size*m, n);	/* size*m, because in MPI_createMatrix this will be divided by size */

		recvcounts = (int*)malloc(size * sizeof(int));
		displs = (int*)malloc(size * sizeof(int));
		for(i=0; i<size-1; i++) recvcounts[i] = _m * n;
		recvcounts[size-1] = (m - _m * (size-1)) * n;
		for(i=0; i<size; i++) displs[i] = i * (_m * n);
	} else {
		tempMatrix = (DT**)malloc(sizeof(DT*));
	}

	MPI_Gatherv(matrix[0], _m*n, MPI_DT, tempMatrix[0], recvcounts, displs, MPI_DT, 0, mycomm);
		//if(rank == 0) printMatrix(tempMatrix, m, n);fflush(stdout);fflush(stderr);

	if(rank == 0) {
		free(displs);
		free(recvcounts);

		for(i=0; i<m; i++) {
			for(j=0; j<n; j++) {
				b[i] += tempMatrix[i][j] * x[j];
			}
		}
	}

	IDL(INFO, printf("\n<----Exit MPI_MatxVec_pruef() for rank=%i\n", rank));

	return b;
}

void MPI_printMatrix(DT ** matrix, int m, int n) {
	int i, j, _m, message;
	MPI_Status status;

	IDL(INFO, printf("\n---->Entered MPI_printMatrix() for rank=%i\n", rank));

	_m = ceil(m / (float)size);
	if(rank+1 == size) {
		_m = m - _m * (size-1);
	}

	if(rank==0) {
		message=1;
		for(i=0; i<_m; i++) {
			for(j=0; j<n-1; j++) {
				printf(FORMAT1, matrix[i][j]);
			}
			printf(FORMAT2, matrix[i][n-1]);
		}
		MPI_Send(&message, 1, MPI_INT, (rank+1) % size, tag, mycomm);
	} else {
		MPI_Recv(&message, 1, MPI_INT, rank-1, tag, mycomm, &status);
		for(i=0; i<_m; i++) {
			for(j=0; j<n-1; j++) {
				printf(FORMAT1, matrix[i][j]);
			}
			printf(FORMAT2, matrix[i][n-1]);
		}
	}

	if(rank==0) {
		MPI_Recv(&message, 1, MPI_INT, (rank+size-1) % size, tag, mycomm, &status);
	} else {
		MPI_Send(&message, 1, MPI_INT, (rank+1) % size, tag, mycomm);
	}

	IDL(INFO, printf("<----Exit MPI_printMatrix() for rank=%i\n", rank));
}


