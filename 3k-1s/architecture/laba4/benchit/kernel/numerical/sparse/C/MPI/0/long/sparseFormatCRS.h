/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatCRS.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/long/sparseFormatCRS.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#ifndef SPARSEFORMATCRS_H
#define SPARSEFORMATCRS_H


#include "sparse.h"

/* Compressed Row Storage */
typedef struct {
	DT * values;
	int sizeOfValues;
	int * columnIndex;
	int sizeOfColumnIndex;
	int * crPointer;
	int sizeOfcrPointer;
} CRS;

extern CRS * MPI_convertToCrsSparse( DT ** matrix, int m, int n);
extern CRS * MPI_gatherCRS( CRS * crs, int m, int n );
extern CRS * MPI_scatterCRS( CRS * crs, int m, int n );
extern DT * MPI_CRSxVec( CRS * crs, int sizeOfB, DT * vec, int sizeOfVec );
extern void MPI_printCrsSparse( CRS * crs );
extern void MPI_clearCrsSparse( CRS * crs );


#endif //SPARSEFORMATCCS_H

