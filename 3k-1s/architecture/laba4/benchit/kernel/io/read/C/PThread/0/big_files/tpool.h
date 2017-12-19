/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: tpool.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/read/C/PThread/0/big_files/tpool.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* the header-file for the PThread-Pool and nothing else
********************************************************************/

#ifndef __iotpool_h
#define __iotpool_h
#endif

#include <pthread.h>

/* this is an standard-interface for threadpools */


/* represents a single request in the requestqueue */
typedef struct tpool_work {
	void               (*routine)();
	void                *arg;
	struct tpool_work   *next;
} tpool_work_t;


/* characteristics and state of a mutexed / synchronized / single thread-pool */	
typedef struct tpool {
	/* pool characteristics */
	int                 num_threads;
        int                 max_queue_size;
        int                 do_not_block_when_full;
        /* pool state */
	pthread_t           *threads;
        int                 cur_queue_size;
	tpool_work_t        *queue_head;
	tpool_work_t        *queue_tail;
	int                 queue_closed;
        int                 shutdown;
	/* pool synchronization */
        pthread_mutex_t     queue_lock;
        pthread_cond_t      queue_not_empty;
        pthread_cond_t      queue_not_full;
	pthread_cond_t      queue_empty;
} *tpool_t;


void tpool_init(
           tpool_t          *tpoolp,
           int              num_threads, 
           int              max_queue_size,
           int              do_not_block_when_full);


int tpool_add_work(
           tpool_t          tpool,
           void             (*routine)(),
	   void             *arg);


int tpool_destroy(
           tpool_t          tpool,
           int              finish);


