/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: sparseFormatCCS.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/sparse/C/0/0/double/sparseFormatCCS.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare of storage formates for sparse matrices
 *******************************************************************/
 
#include "sparse.h"

/* Compressed Column Storage */
typedef struct {
	DT * values;
	long sizeOfValues;
	long * rowIndex;
	long sizeOfRowIndex;
	long * ccPointer;
	long sizeOfccPointer;
} CCS;

extern CCS * convertToCcsSparse( DT ** matrix, long m, long n);
extern DT * CCSxVec( CCS * ccs, long sizeOfB, DT * vec, long sizeOfVec );
extern void printCcsSparse( CCS * ccs );
extern void clearCcsSparse( CCS * ccs );


