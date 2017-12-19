/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures read latency of data located in different cache
 *         levels or memory of certain CPUs.
 *******************************************************************/

#ifndef __work_h
#define __work_h

#include <pthread.h>
#include "arch.h"
#include "mm_malloc.h"

#define KERNEL_DESCRIPTION  "memory latency"
#define CODE_SEQUENCE       "mov memory->register"
#define X_AXIS_TEXT         "data set size [Byte]"
#define Y_AXIS_TEXT_1       "latency [ns]"
#define Y_AXIS_TEXT_2       "latency [cycles]"
#define Y_AXIS_TEXT_3       "counter value/ memory accesses"

#define ALLOC_GLOBAL   0x01
#define ALLOC_LOCAL    0x02

#define HUGEPAGES_OFF  0x01
#define HUGEPAGES_ON   0x02

#define RESTORE_TLB 0x10

#define FLUSH(X)  (1<<(X-1))

/* coherency states */
#define MODE_EXCLUSIVE 0x01
#define MODE_MODIFIED  0x02
#define MODE_INVALID   0x04
#define MODE_SHARED    0x08
#define MODE_OWNED     0x10
#define MODE_FORWARD   0x20
#define MODE_DISABLED  0x00

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
   char* buffer;
   char* cache_flush_area;
   unsigned char NUM_FLUSHES;
   unsigned char NUM_USES;
   unsigned char USE_MODE;
   unsigned int SHARE_CPU;
   unsigned char FLUSH_MODE;
   unsigned char hugepages;
   int settings;
   unsigned long long* tlb_collision_check_array;
   unsigned long long* tlb_tags;
   unsigned long long* page_address;
   int num_threads;
   int max_tlblevel;
   int pagesize;
   int tlb_size;
   int tlb_sets;
   int Eventset;
   int num_events;
   long long *values;
   double *papi_results;
   pthread_t *threads;
   int *thread_comm;
   volatile int ack;
   volatile int done;
   struct threaddata *threaddata;
   cpu_info_t *cpuinfo;	
} mydata_t;

/* data needed by each thread */
typedef struct threaddata
{
   //unsigned long long start_dummy_cachelines[16]; 	//avoid prefetching mydata_t structure when accessing memory 
   volatile mydata_t *data;				//  8 byte
   char* cache_flush_area;				// 16 byte
   cpu_info_t *cpuinfo;					// 24 byte
   char* buffer;                // 32 byte
   unsigned long long* page_address;
   volatile unsigned long long aligned_addr;			// 48 byte
   unsigned int thread_id;
   unsigned int cpu_id;
   unsigned int memsize;
   unsigned int accesses;
   unsigned int settings;				// 68 byte
   unsigned int buffersize;     // 72 byte
   unsigned int alignment;      // 76 byte
   unsigned int offset;         // 80 byte
   unsigned char NUM_FLUSHES;
   unsigned char NUM_USES;
   unsigned char USE_MODE;
   unsigned char FLUSH_MODE;				// 84 byte
   unsigned char fill_cacheline[44];			// 128 byte
   unsigned long long end_dummy_cachelines[16]; 	//avoid prefetching following data 
} threaddata_t;

/*
 * function that does the measurement
 */
extern void _work(int memsize, int def_alignment, int offset, int num_accesses, int runs,volatile mydata_t* data, double ** results);

/*
 * functions for using memory and flushing caches between measurements on different CPUs
 */
extern int cacheflush(int level,int num_flushes,int mode,void* buffer, cpu_info_t cpuinfo);
extern void flush_caches(void* buffer,unsigned long long memsize,int settings,int num_flushes,int flush_mode,void* flush_buffer,cpu_info_t *cpuinfo);

/*
 * use memory before measurement
 */
extern void use_memory(void* buffer,unsigned long long memsize,int rw,int repeat,cpu_info_t cpuinfo,volatile mydata_t *data, threaddata_t *threaddata);

/*
 * loop executed by all threads, except the master thread
 */
extern void *thread(void *threaddata);

#endif
