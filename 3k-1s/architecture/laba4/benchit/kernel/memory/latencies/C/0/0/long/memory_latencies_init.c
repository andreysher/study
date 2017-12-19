/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: memory_latencies_init.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/latencies/C/0/0/long/memory_latencies_init.c $
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


int NUMSTRIDES, nMeasurements;
long *stride, MEM_MIN, MEM_MAX,nCacheSize=0, *measurements;
double calloverhead = 0.0;

void init_(SUMMAND *mem, long *size);
int getnumberofversions(void);
void useversion_(int *version);
extern unsigned long mem_read_b2b(void *ptr, long size, int stride);
extern void sgenrand(unsigned long seed);
extern double genrand(void);
long flushCache(void);

void deallocate(SUMMAND *mem);
double getseqentryoverhead(void *mem);
int mylog(int mun);
void getRndTimeParameters(void* mem);
int playOK(int step, int rank);
double dRndA=0.0, dRndB=0.0;
double dMemFactor=0.0;

char* bi_needs(const char*);

char* bi_needs(const char* env){
	char *p;
	p = bi_getenv(env,0);
	 if (p == NULL) {
		fprintf(stderr, "This kernel needs environment variable: %s\n", env);
		exit(1);
	}
	return p;
}

void bi_getinfo(bi_info* infostruct) {
	int a, i;
	char buff[80], *p;
	float freq=0;

	//printf("\nTimer granularity: %e ns\n", dTimerGranularity*1.0e+9);
	p = bi_needs("BENCHIT_KERNEL_MEMORYLATENCIES_C_READ_STRIDES");
	NUMSTRIDES=1;
	while (p) {
		p = strstr(p,",");
		if (p) {
			p++;
			NUMSTRIDES++;
		}
	}

	stride = (long*)malloc(sizeof(long) * NUMSTRIDES);

	p = bi_needs("BENCHIT_KERNEL_MEMORYLATENCIES_C_READ_STRIDES");
	stride[NUMSTRIDES-1] = atol(p);

	for (i=NUMSTRIDES-2;i>=0; i--) {
		p = strstr(p,",")+1;
		stride[i] = atol(p);
	}

	p = bi_getenv("BENCHIT_KERNEL_MEM_READ_POINTS",0);
	if (p != NULL)  {
		nMeasurements=1;
		while (p) {
			p = strstr(p,",");
			if (p) {
				p++;
				nMeasurements++;
			}
		}
		measurements = (long*)malloc(sizeof(long) * nMeasurements);
		/* if (== NULL) Abbruchkriterium!*/
		p = bi_getenv("BENCHIT_KERNEL_MEM_READ_POINTS",0);
		measurements[0] = atof(p)* 1024*1024;
		MEM_MIN=measurements[0];
		MEM_MAX = measurements[0];

		for (i=1; i < nMeasurements; i++) {
			p = strstr(p,",")+1;
			measurements[i] = atof(p)* 1024*1024;
			if (measurements[i] < MEM_MIN) MEM_MIN=measurements[i];
			if (measurements[i] > MEM_MAX) MEM_MAX=measurements[i];
		}
	}
	else {
		measurements = NULL;
		nMeasurements =atoi(bi_needs("BENCHIT_KERNEL_MEMORYLATENCIES_C_READ_MEASUREMENTS"));
		MEM_MIN =atof(bi_needs("BENCHIT_KERNEL_MEMORYLATENCIES_C_READ_MIN")) * 1024*1024;
		MEM_MAX =atol(bi_needs("BENCHIT_KERNEL_MEMORYLATENCIES_C_READ_MAX")) * 1024*1024;
	}

 	p = bi_getenv("BENCHIT_KERNEL_MEMORYLATENCIES_C_READ_L2CACHE",0);
	if (p == NULL)  {
		nCacheSize = 0;
		p = bi_getenv("BENCHIT_CPU_NAME",0);
		if (p) {
			freq = bi_cpu_freq();
			if (freq>0)
				sprintf(buff,"Memory Latency (%s - %.0lf MHz)", p,1000*freq);
			else
				sprintf(buff,"Memory Latency (%s)", p);
		}
		else
			sprintf(buff,"Memory Latency");
	}
	else {
		nCacheSize = atof(p) * 1024*1024;
		sprintf(buff,"Memory Latency(flush %.1lf MB cache)",((double)nCacheSize)/1024/1024);
 }

    infostruct->kerneldescription = bi_strdup("memory latencies (C)");
    infostruct->codesequence=bi_strdup("do I=1,N#"
				       "  checksum += A[I]#");
    infostruct->xaxistext=bi_strdup("access size");
    infostruct->num_measurements=nMeasurements;
    infostruct->numfunctions= NUMSTRIDES;
		infostruct->base_xaxis = 10;

	/* allocating memory for y axis texts and properties */
  allocYAxis(infostruct);

	for(a=0; a<infostruct->numfunctions; a++) {
		infostruct->yaxistexts[a] = bi_strdup ("latency");
		infostruct->selected_result[a] = SELECT_RESULT_LOWEST;
		infostruct->base_yaxis[a] = 10;
		sprintf(buff, "stride = %d x %d",(int)stride[a],(int)sizeof(SUMMAND));
		infostruct->legendtexts[a] = bi_strdup (buff);
	}
}
/*
double getseqentryoverhead(void *mem) {
  double start, stop;
  int s;

  start=bi_gettime();
  for(s=0; s<10000; s++) {
    mem_read_b2b(mem, 0,1);
  }
  stop=bi_gettime();
  return (stop-start) / 10000;
}
*/

BI_GET_CALL_OVERHEAD_FUNC((void *mem),mem_read_b2b(mem,0,1))

void* bi_init(int problemSizemax) {
	void *mem;
	unsigned long mysize, i;
	SUMMAND *ptr;

	IDL(3, printf("Enter init\n"));

	IDL(-1,printf("%i\n",problemSizemax));
	if (nMeasurements>1) {
		dMemFactor =((double)MEM_MAX)/((double)MEM_MIN);
		dMemFactor = pow(dMemFactor, 1.0/((double)nMeasurements-1));
		mysize=MEM_MAX;
	} else {
		dMemFactor = 1;
		mysize= MEM_MIN;
	}
	IDL(2, printf("allocate size %ld\n", mysize));

	mem = (void*) malloc(mysize);
	if(mem == NULL) {
		printf("Error: malloc (%.2f MB) failed in bi_init()\n",
	    	(double)(mysize)/(double)(1024*1024));
		exit(127);
    }
	i=mysize/sizeof(SUMMAND);
	/* initialize memory with "1" - so that the sum of all cells gives "i" */
	ptr = (SUMMAND*) mem;
	while (i-->0)
		ptr[i] = (SUMMAND)1;

	calloverhead = bi_get_call_overhead(mem);
	printf("\nCall overhead: %.9g ns\n", calloverhead*1.0e+09);
	return mem;
}

extern void bi_cleanup(void *mem) {
    IDL(3, printf("cleaning..."));
    if(mem!=NULL) {
		free(mem);
    }
}
