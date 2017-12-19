/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures aggregate read bandwidth of multiple parallel threads.
 *******************************************************************/

#ifndef __WORK_H
#define __WORK_H

#include <pthread.h>
#include <mm_malloc.h>
#include "arch.h"

#ifdef UNCORE
#include <perfmon/pfmlib.h>
#include <perfmon/pfmlib_intel_nhm.h>
#include <perfmon/perfmon.h>
#endif

#define FLUSH(X)  (1<<(X-1))

#define ALLOC_GLOBAL   0x01
#define ALLOC_LOCAL    0x02

#define HUGEPAGES_OFF  0x01
#define HUGEPAGES_ON   0x02

#define LIFO           0x01
#define FIFO           0x02

/* coherency states */
#define MODE_EXCLUSIVE 0x01
#define MODE_MODIFIED  0x02
#define MODE_INVALID   0x04
#define MODE_SHARED    0x08
#define MODE_OWNED     0x10
#define MODE_FORWARD   0x20
#define MODE_DISABLED  0x00

#define USE_LOAD_PI    	0x01
#define USE_STORE    	 	0x02
#define USE_STORE_NT   	0x03
#define USE_COPY       	0x04
#define USE_COPY_NT    	0x05
#define USE_SCALE_INT  	0x06
#define USE_MUL_PI     	0x07
#define USE_ADD_PI     	0x08
#define USE_MUL_SI     	0x09
#define USE_ADD_SI     	0x0a
#define USE_MUL_PD     	0x0b
#define USE_ADD_PD     	0x0c
#define USE_MUL_SD     	0x0d
#define USE_ADD_SD     	0x0e
#define USE_MUL_PS     	0x0f
#define USE_ADD_PS     	0x10
#define USE_MUL_SS     	0x11
#define USE_ADD_SS     	0x12
#define USE_DIV_PD			0x13
#define USE_DIV_PS			0x14
#define USE_DIV_SD			0x15
#define USE_DIV_SS			0x16
#define USE_SQRT_PD			0x17
#define USE_SQRT_PS			0x18
#define USE_SQRT_SD			0x19
#define USE_SQRT_SS			0x1a
#define USE_AND_PD			0x1b
#define USE_MUL_ADD_PD	0x1c
#define USE_AND_PI	    0x1d
#define USE_LOAD_PD    	0x1e
#define USE_MUL_PLUS_ADD_PD	0x1f
#define USE_LOAD_PS    	0x20

#define REGION_L1 1
#define REGION_L2 2
#define REGION_L3 3
#define REGION_RAM 4

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
   long long *INT_INIT;
   double *FP_INIT;
   unsigned int settings;				// 44 byte
   unsigned short num_threads;				// 46 byte
   unsigned char NUM_FLUSHES;
   unsigned char NUM_USES;
   unsigned char USE_MODE;
   unsigned char FLUSH_MODE;
   unsigned char USE_DIRECTION;
   unsigned char hugepages;				// 52 byte
   unsigned char fill_cacheline1[12];			// 64 byte
   unsigned long long dummy_cachelines[16]; 		// separate exclusive data from shared data
   volatile unsigned long long synch[2]; //16 byte 
   int *thread_comm;					//  24 byte
   volatile unsigned int running_threads; //28 byte  
   volatile unsigned short ack;				// 30 byte
   volatile unsigned short done;			// 32 byte
   long long *values;
   double *papi_results;
   int *threads_per_package;
   int Eventset;
   int num_events;                    //64 byte  
   #ifdef UNCORE
   pfmlib_input_param_t inp;
   pfmlib_nhm_input_param_t mod_inp;
   pfmlib_output_param_t outp;
   pfmlib_event_t *events;
   int pfmon_num_events;
   char **pfm_names;
   int *pfm_codes;
   unsigned int *cid_pfm;
   unsigned int *gid_pfm;   
   #endif
   #ifdef USE_PAPI
   unsigned int *cid_papi;
   unsigned int *gid_papi;
   #endif
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
   unsigned long long length; 
   unsigned long long start_ts;
   unsigned long long end_ts; //64 byte
   unsigned int thread_id;
   unsigned int memsize;      //72 byte
   unsigned int accesses;
   unsigned int settings;				// 80 byte
   unsigned int buffersize;     // 84 byte
   unsigned int alignment;      // 88 byte
   unsigned int offset;         // 92 byte
   unsigned int cpu_id;          // 96
   unsigned int thread_offset;   // 100
   unsigned char NUM_FLUSHES;
   unsigned char NUM_USES;
   unsigned char USE_MODE;
   unsigned char FLUSH_MODE;				// 104 byte
   unsigned char USE_DIRECTION;
   unsigned char FUNCTION;
   unsigned char BURST_LENGTH;
   unsigned char fill_cacheline[21];			// 128 byte
   long long *values;
   double *papi_results;
   int Eventset;
   int num_events;
   int region;
   int package;
   int monitor_uncore;
   #ifdef UNCORE
   pfarg_ctx_t ctx;
   pfarg_pmc_t *pc;
   pfarg_pmd_t *pd;
   pfarg_load_t load_arg;
   int fd;
   #else
   unsigned char fill_cacheline_2[28];			// 128 byte
   #endif
   unsigned long long end_dummy_cachelines[16]; 	//avoid prefetching following data
} threaddata_t;

/*
 * use memory before measurement
 */
extern void use_memory(void* buffer,unsigned long long memsize,int mode,int direction,int repeat,cpu_info_t cpuinfo);

/*
 * function that does the measurement
 */
extern void _work(unsigned long long memsize, int offset, int function, int burst_length,volatile mydata_t* data, double **results);

/*
 * loop executed by all threads, except the master thread
 */
extern void *thread(void *threaddata);

#endif
