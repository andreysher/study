/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatCCS.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/MPI/0/int/sparseFormatCCS.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/

#ifndef SPARSEFORMATCCS_H
#define SPARSEFORMATCCS_H


#include "sparse.h"

/* Compressed Column Storage */
typedef struct {
	DT * values;
	int sizeOfValues;
	int * rowIndex;
	int sizeOfRowIndex;
	int * ccPointer;
	int sizeOfccPointer;
} CCS;

extern CCS * MPI_convertToCcsSparse( DT ** matrix, int m, int n);
extern CCS * MPI_gatherCCS( CCS * ccs, int m, int n );
extern CCS * MPI_scatterCCS( CCS * ccs, int m, int n );
extern DT * MPI_CCSxVec( CCS * ccs, int sizeOfB, DT * vec, int sizeOfVec );
extern void MPI_printCcsSparse( CCS * ccs );
extern void MPI_clearCcsSparse( CCS * ccs );


#endif //SPARSEFORMATCCS_H


