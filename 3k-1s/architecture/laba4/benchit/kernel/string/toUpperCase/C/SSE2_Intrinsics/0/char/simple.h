/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: simple.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/string/toUpperCase/C/SSE2_Intrinsics/0/char/simple.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: SSE String Operations
 *******************************************************************/

#ifndef __work_h
#define __work_h

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata
{
   myinttype offset;
   char * field;
   /* additional parameters */
   myinttype maxsize;
} mydata_t;

extern myinttype toUpperCase( char* field , myinttype size);
extern myinttype toUpperCaseSSE( char* field , myinttype size);
extern myinttype toUpperCaseSSEunaligned( char* field , myinttype size);

#endif

