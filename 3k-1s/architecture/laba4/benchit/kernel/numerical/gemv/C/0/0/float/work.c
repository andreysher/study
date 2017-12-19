/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/gemv/C/0/0/float/work.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: C SGEMV kernel
 *******************************************************************/

#include "work.h"

void ij_(int sizeVector,int sizeAusgabe,float alpha,float beta, float *x, float *A, float *y)
{
	int i,j;
        float temp = 0.0;
	for (j=0;j<sizeAusgabe;j++)
	{
		y[j]=beta*y[j];
	}
	//
	// now : x=x, A=A, y=beta*y
	//

	for (i=0;i<sizeVector;i++)
	{
                temp=alpha*x[i];
		for (j=0;j<sizeAusgabe;j++)
		{
			y[j]=y[j]+A[i*sizeAusgabe+j]*temp;
		}
	}
}

void ji_(int sizeVector,int sizeAusgabe,float alpha,float beta, float *x, float *A, float *y)
{
	int i,j;
        float temp = 0.0;
	for (j=0;j<sizeAusgabe;j++)
	{
		y[j]=beta*y[j];
	}
	//
	// now : x=x, A=A, y=beta*y
	//

	for (j=0;j<sizeAusgabe;j++)
	{
                temp=0.0;
		for (i=0;i<sizeVector;i++)
		{
			temp=temp+A[i*sizeAusgabe+j]*x[i];
		}
                temp=temp*alpha;
                y[j]=y[j]+temp;
	}
}

