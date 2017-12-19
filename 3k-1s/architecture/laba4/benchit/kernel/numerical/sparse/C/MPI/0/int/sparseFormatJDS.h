/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatJDS.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/int/sparseFormatJDS.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#ifndef SPARSEFORMATJDS_H
#define SPARSEFORMATJDS_H


#include "sparse.h"

/* Jagged Diagonal Storage */
typedef struct {
	DT * values;
	int sizeOfValues;
	int * columnIndex;
	int sizeOfColumnIndex;
	int * permutation;
	int sizeOfPermutation;
	int * jdPointer;
	int sizeOfjdPointer;
} JDS;

extern JDS * MPI_convertToJdsSparse( DT ** matrix, int m, int n);
extern JDS * MPI_gatherJDS( JDS * jds, int m, int n );
extern JDS * MPI_scatterJDS( JDS * jds, int m, int n );
extern DT * MPI_JDSxVec( JDS * jds, int sizeOfB, DT * vec, int sizeOfVec );
extern void MPI_printJdsSparse( JDS * jds );
extern void MPI_clearJdsSparse( JDS * jds );

typedef struct {
	int value;
	int oldPosition;
} SortedElem;

extern void sort_QuickSort(SortedElem *left, SortedElem *right);


#endif //SPARSEFORMATJDS_H


