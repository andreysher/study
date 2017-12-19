/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: errors.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/C/PThread/0/1_malloc/errors.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension, for a given number of posix threads,
 *         one malloc for biggest dimension
 *******************************************************************/

#ifndef __errors_h
#define __errors_h

#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EMALLOC 100

#ifndef DEBUG
#define DPRINTF(arg) printf arg
#else
#define DPRINTF(arg)
#endif

#define err_abort(code,text) do { \
    fprintf(stderr, "%s at\"%s\":%d: %s\n", \
	text, __FILE__, __LINE__, strerror(code)); \
    abort(); \
    } while (0);

#define errno_abort(text) do { \
    fprintf(stderr, "%s at\"%s\":%d: %s\n", \
	text, __FILE__, __LINE__, strerror(errno)); \
    abort(); \
    } while (0);

#endif

