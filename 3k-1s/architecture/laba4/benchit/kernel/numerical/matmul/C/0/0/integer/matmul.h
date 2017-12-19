/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matmul.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/0/0/integer/matmul.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: blocked Matrix Multiplication (C)
 *******************************************************************/

#ifndef BENCHIT_MATMUL_H
#define BENCHIT_MATMUL_H

#endif /* BENCHIT_MATMUL_H */

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif


