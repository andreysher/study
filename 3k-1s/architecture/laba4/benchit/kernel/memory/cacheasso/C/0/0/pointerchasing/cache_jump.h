/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: cache_jump.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/cacheasso/C/0/0/pointerchasing/cache_jump.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Kernel to determine the association of the L1 data cache
 *******************************************************************/
 
#ifndef __cache_jump_h
#define __cache_jump_h

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

#define MINTIME 1.0e-22

#define MINSTRIDE_DEFAULT 		 32768
#define MAXSTRIDE_DEFAULT 		 131072
#define MAXLOOPLENGTH_DEFAULT	 20

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata
{
   myinttype minstride;
   myinttype maxstride;
   myinttype maxlooplength;
   void *mem;
} mydata_t;

#define ONE {ptr=(void **) *ptr;}
#define TEN ONE ONE ONE ONE ONE ONE ONE ONE ONE ONE
#define HUN TEN TEN TEN TEN TEN TEN TEN TEN TEN TEN
#define THO HUN HUN HUN HUN HUN HUN HUN HUN HUN HUN

extern void* jump_around( void *mcb, int problemsize);
extern void  make_jump_structure( void *mem, int access_length, int jump_skip );

#endif

