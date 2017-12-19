/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: cache_jump.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/cacheasso/C/0/0/pointerchasing/cache_jump.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Kernel to determine the association of the L1 data cache
 *******************************************************************/

#include <stdio.h>
#include "interface.h"

#include "cache_jump.h"

void *jump_around(void *mem, int problemSize) {
   void **ptr;
   int a;
   
   if(problemSize==0)
      return (void *) 0;	
   
   ptr = (void **)mem;
   
   /* 4.000.000 cycle jumps :-) */
   for (a=0; a<40000; a++) {
      HUN
   }
   return (void *) ptr;
}

void make_jump_structure(void *mem, int access_length, int jump_skip) {
   void **ptr, **first, **nextaddr;
   double numlists, list;
   int a;
   
   IDL(2, printf("ACCESS LENGTH: %d   STRIDE: %d\n", access_length, jump_skip));
   
   ptr = (void **)mem;
   first = ptr;
   /* Anzahl der Listen berechnen, die im Speicher parallel 
    * angelegt werden koennen */
   numlists = (double)(jump_skip/sizeof(void*));
   IDL(2, printf("%f LISTS\n", numlists));
   nextaddr = (void **)mem;
   IDL(3, printf("%p ", (void *)(first)));
   for (list=0.0; list<numlists; list=list+1.0) {
      if(list!=0.0)
         nextaddr+=1;
      ptr=nextaddr;
      IDL(3, printf("%p ", (void *)(nextaddr)));
      for (a=0; a<access_length; a++) {
         if(a==access_length-1) {
            if(list==numlists-1) {
               *ptr = (void *)mem;
            } else {
               *ptr = (void *)nextaddr;
            }
         } else {
            *ptr = (void *)(ptr+jump_skip/sizeof(void*));
            ptr += jump_skip/sizeof(void*);
         }
      }
   }
   IDL(3, printf("\n"));
}

