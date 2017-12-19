/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/bandwidth/C/OpenMP/0/double_stream_opt/work.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measure Bandwidth inspired by STREAM benchmark (C OMP-version)
 *
 * according to the rules, reffer this Benchmark as:
 * "BenchIT kernel based on a variant of the STREAM benchmark code"
 * when publishing results
 *
 * This file contains the work, that is done: copy,scale,add and triad
 *******************************************************************/ 
 
#include "work.h" 

/**
* Copy:
* for all threads:
*   for (j=0;j<repeats)
*     for (i=offset;i<size+offset)
*       alla[thread_nr][i]=allb[thread_nr][i]
* resulting in size*repeats*2*sizeof(double) accessed bytes
**/
double copy_(double **alla, double **allb, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads)
{
  /* stores the measured time */
  double time=0.0;
  /* start parallel work */
	#pragma omp parallel
	{
    /* used for pinning threads */
    long long mask;
    /* used for getting correct data */
		int num,i,k,min,max;
		double *a;
		double *b;
    num=omp_get_thread_num();
    if (localAlloc){
		  a = alla[num];
		  b = allb[num];
	    min=offset;
	    max=size+offset;
    }else{
		  a = alla[0];
		  b = allb[0];
      min=((omp_get_thread_num()*size)/omp_get_num_threads())+offset;
      max=min+size/omp_get_num_threads()+offset-1;
    }
#ifdef BENCHIT_KERNEL_COMPILE_FOR_PIN_THREADS_TO_CORES
    if(pinThreads){
  	  /* pin to correct core */
 	  	mask=1<<num;
 	  	sched_setaffinity(0,sizeof(long long),&mask);
  	  /* done pinning to correct core */
  	}
#endif
    /* take start time */
		#pragma omp barrier
		if (num==0)
			time=bi_gettime();
		#pragma omp barrier
    /* repeat measurement for accuracy */
		for (k=0;k<repeats;k++)
		/* enable aligned access (may increase performance on x86 systems) */
#ifdef BENCHIT_KERNEL_ENABLE_ALIGNED_ACCESS
		#pragma vector aligned
#endif
		/* enable nontemporal stores (may increase performance on x86 systems) */
#ifdef BENCHIT_KERNEL_ENABLE_NONTEMPORAL_STORES
    #pragma vector nontemporal (a)
#endif
		for (i=min;i<max;i++)
		{
			a[i]=b[i];
		}
		#pragma omp barrier
    /* take end time */
		if (num==0)
			time=bi_gettime()-time;
	}
	return time;
}
double scale_(double **alla, double **allb, double scalar, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads)
{
  double time=0.0;
	#pragma omp parallel
	{
    long long mask;
		double *a;
		double *b;
		int num,i,k,min,max;
    num=omp_get_thread_num();
    if (localAlloc){
		  a = alla[num];
		  b = allb[num];
	    min=offset;
	    max=size+offset;
    }else{
		  a = alla[0];
		  b = allb[0];
      min=((omp_get_thread_num()*size)/omp_get_num_threads())+offset;
      max=min+size/omp_get_num_threads()+offset-1;
    }
#ifdef BENCHIT_KERNEL_COMPILE_FOR_PIN_THREADS_TO_CORES
    if(pinThreads){
  	  /* pin to correct core */
 	  	mask=1<<num;
 	  	sched_setaffinity(0,sizeof(long long),&mask);
  	  /* done pinning to correct core */
  	}
#endif
		#pragma omp barrier
		if (num==0)
			time=bi_gettime();
		#pragma omp barrier
		for (k=0;k<repeats;k++)
#ifdef BENCHIT_KERNEL_ENABLE_ALIGNED_ACCESS
		#pragma vector aligned
#endif
#ifdef BENCHIT_KERNEL_ENABLE_NONTEMPORAL_STORES
    #pragma vector nontemporal (a)
#endif
		for (i=min;i<max;i++)
		{
			a[i]=b[i]*scalar;
		}
		#pragma omp barrier
		if (num==0)
			time=bi_gettime()-time;
	}
	return time;
}
double add_(double **alla, double **allb, double **allc, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads)
{
  double time=0.0;
	#pragma omp parallel
	{
    long long mask;
		double *a;
		double *b;
		double *c;
		int num,i,k,min,max;
    num=omp_get_thread_num();
    if (localAlloc){
		  a = alla[num];
		  b = allb[num];
		  c = allc[num];
	    min=offset;
	    max=size+offset;
    }else{
		  a = alla[0];
		  b = allb[0];
		  c = allc[0];
      min=((omp_get_thread_num()*size)/omp_get_num_threads())+offset;
      max=min+size/omp_get_num_threads()+offset-1;
    }
#ifdef BENCHIT_KERNEL_COMPILE_FOR_PIN_THREADS_TO_CORES
    if(pinThreads){
  	  /* pin to correct core */
 	  	mask=1<<num;
 	  	sched_setaffinity(0,sizeof(long long),&mask);
  	  /* done pinning to correct core */
  	}
#endif
		#pragma omp barrier
		if (num==0)
			time=bi_gettime();
		#pragma omp barrier
		for (k=0;k<repeats;k++)
#ifdef BENCHIT_KERNEL_ENABLE_ALIGNED_ACCESS
		#pragma vector aligned
#endif
#ifdef BENCHIT_KERNEL_ENABLE_NONTEMPORAL_STORES
    #pragma vector nontemporal (a)
#endif
		for (i=min;i<max;i++)
		{
			a[i]=b[i]+c[i];
		}
		#pragma omp barrier
		if (num==0)
			time=bi_gettime()-time;
	}
	return time;
}
double triad_(double **alla, double **allb, double **allc, double scalar, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads)
{
  double time=0.0;
	#pragma omp parallel
	{
    long long mask;
		int num,i,k,min,max;
		double *a;
		double *b;
		double *c;
    num=omp_get_thread_num();
    if (localAlloc){
		  a = alla[num];
		  b = allb[num];
		  c = allc[num];
	    min=offset;
	    max=size+offset;
    }else{
		  a = alla[0];
		  b = allb[0];
		  c = allc[0];
      min=((omp_get_thread_num()*size)/omp_get_num_threads())+offset;
      max=min+size/omp_get_num_threads()+offset-1;
    }
#ifdef BENCHIT_KERNEL_COMPILE_FOR_PIN_THREADS_TO_CORES
    if(pinThreads){
  	  /* pin to correct core */
 	  	mask=1<<num;
 	  	sched_setaffinity(0,sizeof(long long),&mask);
  	  /* done pinning to correct core */
  	}
#endif
		#pragma omp barrier
		if (num==0)
			time=bi_gettime();
		#pragma omp barrier
		for (k=0;k<repeats;k++)
#ifdef BENCHIT_KERNEL_ENABLE_ALIGNED_ACCESS
		#pragma vector aligned
#endif
#ifdef BENCHIT_KERNEL_ENABLE_NONTEMPORAL_STORES
    #pragma vector nontemporal (a)
#endif
		for (i=min;i<max;i++)
		{
			a[i]=b[i]*scalar+c[i];
		}
		#pragma omp barrier
		if (num==0)
			time=bi_gettime()-time;
	}
	return time;
}
 

