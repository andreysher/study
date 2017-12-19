/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: errors.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/dotproduct/C/PThread/0/double/errors.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for dot product of two vectors with posix threads
 *******************************************************************/

#ifndef __errors_h
#define __errors_h

#include <errno.h>
#include <pthread.h>

extern char *error_text(int errnum);

#endif

