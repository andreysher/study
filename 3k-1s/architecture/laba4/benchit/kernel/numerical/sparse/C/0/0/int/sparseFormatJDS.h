/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatJDS.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/0/0/int/sparseFormatJDS.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#include "sparse.h"

/* Jagged Diagonal Storage */
typedef struct {
	DT * values;
	long sizeOfValues;
	long * columnIndex;
	long sizeOfColumnIndex;
	long * permutation;
	long sizeOfPermutation;
	long * jdPointer;
	long sizeOfjdPointer;
} JDS;

/*
this struct helps to sort the elements, and save the old position of them
*/
typedef struct {
	long value;
	long oldPosition;
} SortedElem;

extern JDS * convertToJdsSparse( DT ** matrix, long m, long n);
extern DT * JDSxVec( JDS * jds, DT * vec, long sizeOfVec );
extern void printJdsSparse( JDS * jds );
extern void clearJdsSparse( JDS * jds );

extern void sort_QuickSort(SortedElem *left, SortedElem *right);


