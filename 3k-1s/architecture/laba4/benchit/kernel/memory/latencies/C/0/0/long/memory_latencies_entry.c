/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: memory_latencies_entry.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/latencies/C/0/0/long/memory_latencies_entry.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: memory latencies (C)
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "interface.h"
#include "memory_latencies.h"


extern int NUMSTRIDES, nMeasurements;
extern long *stride, MEM_MIN, MEM_MAX,nCacheSize, *measurements;

int mem_read_b2b(SUMMAND *start, int size, int stride);

long flushCache(void);
extern double dMemFactor;
int mylog(int mun);
int confuseCache(int nCacheSize);
int playOK(int step, int rank);

extern char* bi_needs(const char*);
extern double calloverhead;
SUMMAND checksum;

int bi_entry(void *mcb, int problemSize,double *results)
{
    double time;
    double start, stop;
    long mysize, wSize, checksum=0;
	int i;
	if (results == NULL) {
		printf("No mem for results");
		exit(127);
	}
	IDL(2, printf("Start\n"));
    if (measurements == NULL) {
			mysize = (long)(((double)MEM_MIN)*pow(dMemFactor, (problemSize-1)));
		}
		else
			mysize = measurements[problemSize-1];

		wSize = mysize/sizeof(SUMMAND);
		results[0]=(double)mysize;

		IDL(2, printf("run\n"));

		checksum = confuseCache(nCacheSize);
		if (checksum==0)
				printf("Error confusing cache");

	  for (i=0; i<NUMSTRIDES; i++) {
			if (mysize > stride[i] * sizeof (SUMMAND)) {

				checksum = confuseCache(nCacheSize);
				if (checksum==0)
					printf("Error confusing cache");

				start= bi_gettime();
				checksum = mem_read_b2b((SUMMAND*)mcb, wSize, stride[i]);
				stop = bi_gettime();
				time = stop-start - dTimerOverhead - calloverhead;
				checksum = (int)wSize/stride[i];

				if (time < 5*dTimerGranularity) {
					results[i+1]=INVALID_MEASUREMENT;
				}
				else
					results[i+1]=time/checksum;
			}
		}
	IDL(2, printf("result is %f\n", results[0]));
	return 0;
}

int confuseCache(int nCacheSize) {
/* trying to fill the L2-cache with uninteristing stuff */
	int s=0, i,*memConfuse;

	if (nCacheSize == 0)
		return 1;
	memConfuse = (int*)malloc(nCacheSize);
	nCacheSize = nCacheSize/sizeof(int);
	for (i=0; i<nCacheSize; memConfuse[i++]=1);
	for (i = nCacheSize/2; i < nCacheSize; i++)
			s += memConfuse[i]+memConfuse[i-nCacheSize/2];
	for (i = nCacheSize/2; i < nCacheSize; i++)
			s += memConfuse[i]+memConfuse[i-nCacheSize/2];
	for (i = nCacheSize/2; i < nCacheSize; i++)
			s += memConfuse[i]+memConfuse[i-nCacheSize/2];
	for (i = nCacheSize/2; i < nCacheSize; i++)
			s += memConfuse[i]+memConfuse[i-nCacheSize/2];
	free(memConfuse);
	return s;
}

int mem_read_b2b(SUMMAND *start, int size, int stride)
/* Measurement of the back-to-back-latency i.e. the following address
*  can not be calculated befor the current mem contents is known */
{
	register int i;

	if (stride >1) {
		for (i = 0; i < size; i += stride)
			i += start[i]; /* incrementing by 1 */
		return i;
	}
	/* else */
	for (i = 0; i < size; i ++)
		i += start[i];
	return i;
}


