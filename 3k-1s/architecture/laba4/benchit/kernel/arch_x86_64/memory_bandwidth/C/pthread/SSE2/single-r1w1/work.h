/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures combined bandwidth of one read and one write stream located
 *         in different cache levels or memory of certain CPUs.
 *******************************************************************/

#ifndef __WORK_H
#define __WORK_H

#include "arch.h"

#define USE_MOV            0x01
#define USE_MOVNTI         0x02
#define USE_MOVAPD         0x04
#define USE_MOVUPD         0x08
#define USE_MOVNTPD        0x10
#define USE_MOV_CLFLUSH    0x20

#define METHOD_COPY        0x01
#define METHOD_SCALE       0x02
#define METHOD_INDEP       0x04

#define LAYOUT_CONT        0x01
#define LAYOUT_FIXED       0x02
#define LAYOUT_ALT1        0x04
#define LAYOUT_ALT2        0x08

#define STRIDE 64

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
   char* buffer;					        //  8 byte
   char* cache_flush_area;				// 16 byte
   pthread_t *threads;					  // 24 byte
   struct threaddata *threaddata; // 32 byte
   cpu_info_t *cpuinfo;					  // 40 byte
   unsigned long long buffersize; // 48 byte
   double factor;
   double init_value;             // 64 byte
   unsigned int settings;				  // 68 byte
   unsigned int SHARE_CPU;
   unsigned int alignment;        // 76 byte
   unsigned short num_threads;	  // 78 byte
   unsigned char NUM_FLUSHES;
   unsigned char NUM_USES;        // 80 byte
   unsigned char USE_MODE_R;
   unsigned char USE_MODE_W;
   unsigned char FLUSH_MODE;
   unsigned char hugepages;
   unsigned char USE_DIRECTION;
   unsigned char method;
   unsigned char layout;
   unsigned char read_local;      // 88 byte
   unsigned char write_local;
   unsigned char offset;
   unsigned char burst_length;
   unsigned char function;
   unsigned char runs;            // 93 byte
   #ifdef USE_PAPI
   unsigned char papi_dummy[3];   // 96 byte
   long long *values;
   double *papi_results;          // 112 byte
   int Eventset;
   int num_events;                // 120 byte
   unsigned char papi_fill_cacheline[8]; // 128 byte
   #else
   unsigned char fill_cacheline1[35];			// 128 byte    
   #endif 
   unsigned long long dummy_cachelines[16];// separate exclusive data from shared data
   volatile unsigned long long synch;     //  8 byte 
   volatile unsigned long long syncl;     // 16 byte 
   int *thread_comm;					            // 24 byte
   volatile unsigned int running_threads; // 28 byte  
   volatile unsigned short ack;				    // 30 byte
   volatile unsigned short done;			    // 32 byte
   unsigned char fill_cacheline2[32];			// 64 byte
   unsigned long long end_dummy_cachelines[16]; 	// avoid prefetching other memory when accessing mydata_t structure
} mydata_t;

/* data needed by each thread */
typedef struct threaddata
{
   //unsigned long long start_dummy_cachelines[16]; 	//avoid prefetching when accessing other memory 
   volatile mydata_t *data;				  //  8 byte
   char* cache_flush_area;				  // 16 byte
   char* buffer;									  // 24 byte
   cpu_info_t *cpuinfo;					    // 32 byte	
   unsigned long long aligned_addr;	// 40 byte
   unsigned long long start_ts;     
   unsigned long long end_ts;       // 56 byte
   unsigned int thread_id;
   unsigned int memsize;            // 64 byte
   unsigned int accesses;
   unsigned int settings;				    // 72 byte
   unsigned int buffersize;
   unsigned int alignment;          // 80 byte
   unsigned int offset;
   unsigned int cpu_id;             // 88 byte
   unsigned char NUM_FLUSHES;
   unsigned char NUM_USES;
   unsigned char USE_MODE_R;
   unsigned char USE_MODE_W;
   unsigned char FLUSH_MODE;
   unsigned char USE_DIRECTION;     // 94 byte
   unsigned char fill_cacheline[34];			// 128 byte
   unsigned long long end_dummy_cachelines[16]; 	//avoid prefetching following data 
} threaddata_t;


typedef struct parameters param_t;
struct parameters
{
  //unsigned long long start_dummy_cachelines[16]; 	//avoid prefetching when accessing other memory
  void *flushaddr;
  param_t *thread_params;
  param_t *share_cpu_params;
  unsigned long long addr_1;
  unsigned long long addr_2;
  unsigned long long rax;      // 48 byte
  unsigned long long passes;
  unsigned long long memsize;
  double value;
  unsigned long long tmax;     // 80 byte
  double factor;
  int thread_id;
  int i;
  int j;
  unsigned int runs;           // 104 byte
  unsigned int iter;
  unsigned int alignment;
  unsigned int flushsize;      // 116 byte
  unsigned char use_mode_1;
  unsigned char use_mode_2;
  unsigned char default_use_mode_1;
  unsigned char default_use_mode_2;   // 120 byte
  unsigned char use_direction;
  unsigned char num_uses;
  unsigned char layout;
  unsigned char accesses_per_loop;    // 124 byte
  unsigned char num_flushes;
  unsigned char flush_mode;
  unsigned char read_local;
  unsigned char write_local;          // 128 byte
  #ifdef USE_PAPI
  long long* values;
  int Eventset;
  int num_events;                     // 16 byte
  char papi_fill_cacheline[48];
  #endif
  unsigned long long end_dummy_cachelines[16]; 	//avoid prefetching following data 
};

/*
 * use memory before measurement
 */
extern void use_memory(param_t *params);

/*
 * function that does the measurement
 */
extern void _work(unsigned long long memsize,volatile mydata_t* data, double **results);

#endif
