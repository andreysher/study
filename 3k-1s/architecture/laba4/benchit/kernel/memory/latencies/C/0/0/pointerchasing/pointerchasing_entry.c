/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: pointerchasing_entry.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/latencies/C/0/0/pointerchasing/pointerchasing_entry.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Memory Access Time (C)
 *******************************************************************/

#include "interface.h"
#include "pointerchasing.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define ONE {ptr=(void **) *ptr;}
#define TEN ONE ONE ONE ONE ONE ONE ONE ONE ONE ONE
#define HUN TEN TEN TEN TEN TEN TEN TEN TEN TEN TEN
#define THO HUN HUN HUN HUN HUN HUN HUN HUN HUN HUN


extern long numjumps;

void *jump_around(void *mem, long n);


/** generates a random number between 0 and (max-1)
 *  @param  max maximum random number
 *  @return a random number between 0 and (max-1)
 */
unsigned int random_number(unsigned long max)
{
  return (unsigned int) (((double)max)*rand()/(RAND_MAX+1.0));
}


/** creates a memory are that is randomly linked 
 *  @param mem     the memory area to be used
 *  @param length  the number of bytes that should be used
 */
void make_linked_memory(void *mem, long length) {

  /* some pointers to generate the list */
  void **ptr, **first;
  /** how many ptr we create within the memory */
  long num_ptr=length/sizeof(void *);
  /** the list for all memory locations that are linked */
  long *ptr_numbers;
  /** for the loops */
  long loop_ptrs;
  /** actual random number */
  long act_num;

  /* allocate memory for ptr numbers */
  ptr_numbers=(long *) malloc(num_ptr*sizeof(long));
  if(num_ptr>0 && ptr_numbers==NULL)
    {
      printf("no more core in make_linked_mem()\n");
      bi_cleanup(mem);
      exit(1);
    }
  /* initialize ptr numbers, the 0 is used as the first
   * number
   */
  for(loop_ptrs=1; loop_ptrs<num_ptr; loop_ptrs++)
    ptr_numbers[loop_ptrs-1]=loop_ptrs;

  /* init first ptr with first memory location */
  ptr=(void **)mem;
  first=ptr;
   
  num_ptr--;

  while(num_ptr>1) {
    /* get a random position within the
       remaining list */
    act_num=random_number(num_ptr);
    /* create a link from the last ptr 
       to this ptr */
    *ptr=(void *) (first+ptr_numbers[act_num]);
    /* move pointer to new memory location */
    ptr=first+ptr_numbers[act_num];
    /* remove used ptr number from list of
       pointer numbers, just copies the last 
       number to the actual position */
    ptr_numbers[act_num]=ptr_numbers[num_ptr-1];
    num_ptr--;
  }

  /* the last number is linked to the first */
  *ptr=(void *) first;

  /* free the ptr list */
  free(ptr_numbers);
  IDL(4,printf("\n"));
}


int bi_entry(void *mcb,int problemSize,double *results) {

	static double timeroh=0, calloh=0;
	double start, stop;
	int i;

	long length;
	long long c[4];
	void *ptr;
	
	extern double dMemFactor;
	extern long minlength;

	length = (long)(((double)minlength)*pow(dMemFactor, (problemSize-1)));

	results[0]=(double) length;


	IDL(2, printf("Making structure\n"));
	make_linked_memory(mcb, length);
	IDL(2, printf("Enter measurement\n"));
	
	ptr = jump_around(mcb, numjumps); 	
	start=bi_gettime();
	if ((long)ptr == 0)
		printf("#");
	
 	
	jump_around(mcb, numjumps);
 	
	stop=bi_gettime();
	
	IDL(2, printf("Done\n"));
		
	 results[1]=(double)(stop-start)/((double)numjumps);
   

  return (0);
}


