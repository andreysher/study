/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: iowrite.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/write/C/PThread/0/iowrite_c_pthreads/iowrite.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#ifndef __iowrite_h
#define __iowrite_h
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <unistd.h>


/*
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
*/

/****c* io_write_pthreads/types
 * SYNOPSIS
 * this kernel uses a set of selfdefined types.
 * follow the links for further information
 * on each of them
 ***/
 
 
/****f* types/environment_variables_t
 * SYNOPSIS 
 * this type holds all environment variables
 * SOURCE
 */
typedef struct 
	{
	  double FILESIZE;
	  int NUMCHANNELS;
	  int CHANNELFACTOR;
	  char * DISKPATH;
	  double DISKSPACE;
//	  double RAMSIZE;
	  int TIMELIMIT;
	} environment_variables_t;

/*******/	

/****f* types/backpack_t
 * SYNOPSIS 
 * holds all values needed throughout the measurement
 * this includes:
 *   - the environment variables
 *   - ...
 * this is the type of the bi_init-return-value
 * SOURCE
 */
typedef struct
	{
	  environment_variables_t * env_var;
	  char * filepath;
	  char * filebuffer;
	} backpack_t;
/*******/	

	
/****f* types/thread_arg_wrapper_t
 * SYNOPSIS 
 * this is the wrapper-type for the pthreadpool 
 * SOURCE
 */
typedef struct 
	{
	backpack_t * bp;
	unsigned long i;
	double * start_time;
	double * end_time;
	} thread_arg_wrapper_t;


