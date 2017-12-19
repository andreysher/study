/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/bandwidth/C/OpenMP/0/double_stream/work.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measure Bandwidth inspired by STREAM benchmark (C OMP-version)
 *
 *         according to the rules, reffer this Benchmark as:
 *         "BenchIT kernel based on a variant of the STREAM benchmark code"
 *         when publishing results.
 *
 *         This file contains the work, that is done: copy,scale,add
 *         and triad
 *******************************************************************/

#include "work.h" 
 
void copy_(double *a, double *b, int size)
{
	register int i=0;
	#pragma omp parallel for
	for (i=0;i<size;i++)
	{
		a[i]=b[i];
	}
}
void scale_(double *a, double *b, double scalar, int size)
{
	register int i=0;
	#pragma omp parallel for
	for (i=0;i<size;i++)
	{
		a[i]=scalar*b[i];
	}
}
void add_(double *a, double *b, double *c, int size)
{
	register int i=0;
	#pragma omp parallel for
	for (i=0;i<size;i++)
	{
		a[i]=b[i]+c[i];
	}
}
void triad_(double *a, double *b, double *c, double scalar, int size)
{
	register int i=0;
	#pragma omp parallel for
	for (i=0;i<size;i++)
	{
		a[i]=b[i]+scalar*c[i];
	}
}
 

