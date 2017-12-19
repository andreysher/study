/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatCRS.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/0/0/long/sparseFormatCRS.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/


#include "sparse.h"

/* Compressed Row Storage */
typedef struct {
	DT * values;
	long sizeOfValues;
	long * columnIndex;
	long sizeOfColumnIndex;
	long * crPointer;
	long sizeOfcrPointer;
} CRS;

extern CRS * convertToCrsSparse( DT ** matrix, long m, long n);
extern DT * CRSxVec( CRS * crs, DT * vec, long sizeOfVec );
extern void printCrsSparse( CRS * crs );
extern void clearCrsSparse( CRS * crs );


