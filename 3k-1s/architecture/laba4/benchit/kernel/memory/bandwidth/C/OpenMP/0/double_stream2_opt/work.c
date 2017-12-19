#include "work.h"
#include <stdio.h>
#include <inttypes.h>
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
		int num,i,k;
		unsigned long long min,max;
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
double sum_(double **alla, double *result, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads)
{
  double time=0.0;
  double sum;
	#pragma omp parallel reduction(+:sum)
	{
    long long mask;
		double *a;
		int num,i,k;
		unsigned long long min,max;
    num=omp_get_thread_num();
    if (localAlloc){
		  a = alla[num];
	    min=offset;
	    max=size+offset;
    }else{
		  a = alla[0];
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
			sum+=a[i];
		}
		#pragma omp barrier
		if (num==0)
			time=bi_gettime()-time;
	}
	result[0]=sum;
	return time;
}
double fill_(double **alla, double scalar, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads)
{
  double time=0.0;
	#pragma omp parallel
	{
    long long mask;
		double *a;
		int num,i,k;
		unsigned long long min,max;
    num=omp_get_thread_num();
    if (localAlloc){
		  a = alla[num];
	    min=offset;
	    max=size+offset;
    }else{
		  a = alla[0];
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
			a[i]=scalar;
		}
		#pragma omp barrier
		if (num==0)
			time=bi_gettime()-time;
	}
	return time;
}
double daxpy_(double **alla, double **allb, double scalar, unsigned long long size, int offset, long long repeats, int localAlloc, int pinThreads)
{
  double time=0.0;
	#pragma omp parallel
	{
    long long mask;
		int num,i,k;
		unsigned long long min,max;
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
			a[i]=a[i]*scalar+b[i];
		}
		#pragma omp barrier
		if (num==0)
			time=bi_gettime()-time;
	}
	return time;
}
