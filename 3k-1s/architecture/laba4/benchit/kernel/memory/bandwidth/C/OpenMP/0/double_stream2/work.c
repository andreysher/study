#include "work.h"

void copy_(double *a, double *b, int size)
{
	register long long i=0;
	#pragma omp parallel for
	for (i=0;i<size;i++)
	{
		a[i]=b[i];
	}
}
void fill_(double *a, double scalar, int size)
{
	register long long i=0;
	#pragma omp parallel for
	for (i=0;i<size;i++)
	{
		a[i]=scalar;
	}
}
void daxpy_(double *a, double q, double *b, int size)
{
	register long long i=0;
	#pragma omp parallel for
	for (i=0;i<size;i++)
	{
		a[i]=a[i]+q*b[i];
	}
}
double sum_(double *a, int size)
{
	register long long i=0;
	double sum=0.0;
	#pragma omp parallel for reduction(+:sum)
	for (i=0;i<size;i++)
	{
		sum+=a[i];
	}
	return sum;
}
