/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: pointerchasing_accesses.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/latencies/C/0/PAPI/pointerchasing/pointerchasing_accesses.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#define ONE {ptr=(void **) *ptr;}
#define TEN ONE ONE ONE ONE ONE ONE ONE ONE ONE ONE
#define HUN TEN TEN TEN TEN TEN TEN TEN TEN TEN TEN
#define THO HUN HUN HUN HUN HUN HUN HUN HUN HUN HUN

void *jump_around(void *mem, long n) {
  void **ptr;
  long a;


  ptr=(void **) mem;

  /* numjump Sprnge im Kreis :-) */
  for(a=0; a<n/100; a++) {
    HUN
      }
  return (void *) ptr;
}

