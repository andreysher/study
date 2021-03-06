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
 /* identical functions for all single threaded kernels 
  * TODO: migrate changes from r1w1 kernel */
 
#ifndef __SHARED_H
#define __SHARED_H

#include <mm_malloc.h>
#include "arch.h"
#include "work.h"

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

/*
 * functions for using memory and flushing caches between measurements on different CPUs
 */
extern int cacheflush(int level,int num_flushes,int mode,void* buffer, cpu_info_t cpuinfo);
extern void flush_caches(void* buffer,unsigned long long memsize,int settings,int num_flushes,int flush_mode,void* flush_buffer,cpu_info_t *cpuinfo);

/*
 * loop executed by all threads, except the master thread
 */
extern void *thread(void *threaddata);

#endif
