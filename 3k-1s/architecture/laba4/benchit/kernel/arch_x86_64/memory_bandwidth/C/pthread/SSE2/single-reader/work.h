/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures read bandwidth of data located in different cache
 *         levels or memory of certain CPUs.
 *******************************************************************/

#ifndef __WORK_H
#define __WORK_H

#include <pthread.h>
#include "arch.h"

#define USE_MOV        0x01
#define USE_MOVDQA     0x04
#define USE_MOVDQU     0x08

#define THREAD_USE_MEMORY  1
#define THREAD_WAIT        2
#define THREAD_FLUSH       3
#define THREAD_WORK        4
#define THREAD_INIT        5
#define THREAD_STOP        6

/** The data structure that holds all the global data.
 */
typedef struct mydata
{
   //unsigned long long start_dummy_cachelines[16]; 	// avoid prefetching mydata_t structure when accessing other memory 
   char* buffer;					//  8 byte
   char* cache_flush_area;				// 16 byte
   pthread_t *threads;					// 24 byte
   struct threaddata *threaddata;			// 32 byte
   cpu_info_t *cpuinfo;					// 40 byte
   unsigned int settings;				// 44 byte
   unsigned int SHARE_CPU;
   unsigned short num_threads;				// 50 byte
   unsigned char NUM_FLUSHES;
   unsigned char NUM_USES;
   unsigned char USE_MODE;
   unsigned char FLUSH_MODE;
   unsigned char hugepages;				// 55 byte
   unsigned char USE_DIRECTION;
   unsigned char fill_cacheline1[8];			// 64 byte
   unsigned long long dummy_cachelines[16]; 		// separate exclusive data from shared data
   volatile unsigned long long synch; //8 byte 
   volatile unsigned long long syncl; //16 byte 
   int *thread_comm;					//  24 byte
   volatile unsigned int running_threads; //28 byte  
   volatile unsigned short ack;				// 30 byte
   volatile unsigned short done;			// 32 byte
   long long *values;
   double *papi_results;
   int Eventset;
   int num_events;
   unsigned char fill_cacheline2[8];			// 64 byte   
   unsigned long long end_dummy_cachelines[16]; 	// avoid prefetching other memory when accessing mydata_t structure
} mydata_t;

/* data needed by each thread */
typedef struct threaddata
{
   //unsigned long long start_dummy_cachelines[16]; 	//avoid prefetching mydata_t structure when accessing memory 
   volatile mydata_t *data;				//  8 byte
   char* cache_flush_area;				// 16 byte
   char* buffer;									// 24 byte
   cpu_info_t *cpuinfo;					  // 32 byte	
   unsigned long long aligned_addr;			// 40 byte
   unsigned long long start_ts;
   unsigned long long end_ts; //56 byte
   unsigned int thread_id;
   unsigned int memsize;      //64 byte
   unsigned int accesses;
   unsigned int settings;				// 72 byte
   unsigned int buffersize;     // 76 byte
   unsigned int alignment;      // 80 byte
   unsigned int offset;         // 84 byte
   unsigned int cpu_id;          // 88
   unsigned char NUM_FLUSHES;
   unsigned char NUM_USES;
   unsigned char USE_MODE;
   unsigned char FLUSH_MODE;				// 92 byte
   unsigned char USE_DIRECTION;
   unsigned char fill_cacheline[35];			// 128 byte
   unsigned long long end_dummy_cachelines[16]; 	//avoid prefetching following data 
} threaddata_t;

/*
 * use memory before measurement
 */
extern void use_memory(void* buffer,unsigned long long memsize,int mode,int direction,int repeat,cpu_info_t cpuinfo);

/*
 * function that does the measurement
 */
extern void _work(unsigned long long memsize, int offset, int function, int burst_length, int runs,volatile mydata_t* data, double **results);

#endif
