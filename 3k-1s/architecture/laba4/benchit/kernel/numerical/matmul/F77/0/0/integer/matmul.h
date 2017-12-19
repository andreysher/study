/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matmul.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/F77/0/0/integer/matmul.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix Multiply (F77)
 *******************************************************************/

#ifndef BENCHIT_MATMUL_H
#define BENCHIT_MATMUL_H

/* some Fortran compilers export symbols in s special way:
 * all letter are big letters
 */
/*     defined (_SX)      || \   (the SX does this not any longer)*/
#if (defined (_CRAY)    || \
     defined (_SR8000)  || \
     defined (_USE_OLD_STYLE_CRAY_TYPE))
#define multaijk_	MULTAIJK
#define multaikj_	MULTAIKJ
#define multajik_	MULTAJIK
#define multajki_	MULTAJKI
#define multakij_	MULTAKIJ
#define multakji_	MULTAKJI
#endif

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

extern void multaijk_( myinttype *a, myinttype *b, myinttype *c, myinttype *size);
extern void multaikj_( myinttype *a, myinttype *b, myinttype *c, myinttype *size);
extern void multajik_( myinttype *a, myinttype *b, myinttype *c, myinttype *size);
extern void multajki_( myinttype *a, myinttype *b, myinttype *c, myinttype *size);
extern void multakij_( myinttype *a, myinttype *b, myinttype *c, myinttype *size);
extern void multakji_( myinttype *a, myinttype *b, myinttype *c, myinttype *size);

#endif /* BENCHIT_MATMUL_H */


