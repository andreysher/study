/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: memjump.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/latencies/C/MPI/0/pointerchasing/memjump.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Memory Access Time (C)
 *******************************************************************/

#include "interface.h"
#include "memacc.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

#define ONE {ptr=(void **) *ptr;}
#define TEN ONE ONE ONE ONE ONE ONE ONE ONE ONE ONE
#define HUN TEN TEN TEN TEN TEN TEN TEN TEN TEN TEN
#define THO HUN HUN HUN HUN HUN HUN HUN HUN HUN HUN

void *jump_around(void *mem, int problemSize, long numjumps) {
  void **ptr;
  int a;

  if(problemSize==0)
    return (void *) 0;	

  ptr=(void **) mem;

  /* numjump Spruenge im Kreis :-) */
  for(a=0; a<numjumps/100; a++) {
    HUN
      }
  return (void *) ptr;
}


