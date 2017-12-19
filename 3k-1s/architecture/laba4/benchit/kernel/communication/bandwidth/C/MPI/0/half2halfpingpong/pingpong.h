/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: pingpong.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/bandwidth/C/MPI/0/half2halfpingpong/pingpong.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: pairwise Send/Recv between two MPI-Prozesses>
 *         this file holds all the functions needed by the 
 *         benchit-interface
 *******************************************************************/

#ifndef __pingpong_h
#define __pingpong_h

#include "interface.h"
#include "mpi.h"

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 BENCHIT_KERNEL_REPETITIONS=1
 BENCHIT_KERNEL_SHOW_PAIR_BANDWITH="1"
 BENCHIT_KERNEL_SHOW_TOTAL_BANDWITH=1
 BENCHIT_KERNEL_MIN_MSG_SIZE=$((1024*1024*1))
 BENCHIT_KERNEL_MAX_MSG_SIZE=$((1024*1024*1024))
 BENCHIT_KERNEL_MSG_SIZE_INCREMENT=$((1024*1024*1))
 BENCHIT_KERNEL_SENDERLIST=""
 BENCHIT_KERNEL_RECEIVERLIST=""
 */
typedef struct mydata
{
   int commsize;
   int commrank;
   unsigned int repeat;
   unsigned long int pair_bandwith;
   unsigned long int total_bandwith;
   unsigned char empty_list;
   unsigned int * senderlist;
   unsigned int * receiverlist;
   unsigned int * msg_string;
   /* additional parameters */
   unsigned int maxsize;
} mydata_t;

extern void pingpong( unsigned int *from, unsigned int *to, void *mdpv, unsigned long int * msgsize );

#endif


