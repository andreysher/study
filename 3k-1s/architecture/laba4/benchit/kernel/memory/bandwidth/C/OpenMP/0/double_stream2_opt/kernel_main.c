
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/*  Header for local functions
 */
#include "work.h"

/* aligned memory allocation */
#include <errno.h>
#include <stddef.h> /* ptrdiff_t */
#include <stdint.h> /* uintptr_t */

#define NOT_POWER_OF_TWO(n) (((n) & ((n) - 1)))
#define UI(p) ((uintptr_t) p)

/*  Header for local functions
 */
void evaluate_environment(void);
void *_aligned_malloc(size_t size, size_t alignment);
void _aligned_free(void *memblock);
/*
  aligned alloc/free
*/

static void *ptr_align(void *p0, size_t alignment, size_t offset)
{
     return (void *) (((UI(p0) + (alignment + sizeof(void*)) + offset)
		       & (~UI(alignment - 1)))
		       - offset);
}


void *_aligned_offset_malloc(size_t size, size_t alignment, size_t offset)
{
    void *p0, *p;
    if (NOT_POWER_OF_TWO(alignment)) {
        errno = EINVAL;
	return((void*) 0);
    }
    if (size == 0)
	return((void*) 0);
    if (alignment < sizeof(void *))
	alignment = sizeof(void *);
/* including the extra sizeof(void*) is overkill on a 32-bit
    machine, since malloc is already 8-byte aligned, as long
    as we enforce alignment >= 8 ...but oh well */
    p0 = (void*) malloc(size + (alignment + sizeof(void*)));
    if (!p0)
	return((void*) 0);
    p = ptr_align(p0, alignment, offset);
    *(((void **) p) - 1) = p0;
    return p;
}

void *_aligned_malloc(size_t size, size_t alignment)
{
    return _aligned_offset_malloc(size, alignment, 0);
}

void _aligned_free(void *memblock)
{
    if (memblock)
	free(*(((void **) memblock) - 1));
}

/**
* this defines the number of works. There'll be 4 copy, scale, add, triad
**/

int functionCount=4;

/**
* minlength: minimal length to measure (defined in parameters)
* maxlength: maximal length to measure (defined in parameters)
* accessstride: used to calculate the problemSizes between
* internal_repeats: additional repeats (defined in parameters,
*                   multiplies with BENCHIT_ACCURACY)
* offset: offset for memory access (defined in parameters)
* alignment: alignment for memory (defined in parameters)
* nMeasurements: number of problemSizes measured (defined in parameters)
* localAlloc: whether to handle memory local to threads
* threadPinning: whether to pin threads to cores
**/

unsigned long long minlength, maxlength;
long accessstride,internal_repeats,offset,alignment,
     nMeasurements, localAlloc, threadPinning;

/**
* dMemFactor: used to calculate the problemSizes
* minTimeForOneMeasurement: minimal time allowed for one benchmark
*                           (defined in parameters)
**/

double dMemFactor,minTimeForOneMeasurement;


/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
  char buf[200];
  char buf2[200];
   int i = 0; /* loop var for functionCount */
   /* get environment variables for the kernel */
   evaluate_environment();

   sprintf(buf2,"STREAM inspired benchmark (C+OpenMP)#"
               "OFFSET=%i#"
               "ALIGNMENT=%i#"
               "NTIMES=%i#"
               "THREADS=%i#"
#ifdef BENCHIT_KERNEL_ENABLE_ALIGNED_ACCESS
		           "pragma vector aligned enabled#"
#else
		           "pragma vector aligned disabled#"
#endif
#ifdef BENCHIT_KERNEL_ENABLE_NONTEMPORAL_STORES
		           "pragma vector nontemporal enabled#"
#else
		           "pragma vector nontemporal disabled#"
#endif
               ,
               (int)offset,
               (int)alignment,
               (int)internal_repeats,
               (int)omp_get_max_threads());


				if (localAlloc){
					sprintf(buf,"%slocal alloc#",buf2);
				} else{
					sprintf(buf,"%sglobal alloc#",buf2);
				}
				if (threadPinning){
					sprintf(buf2,"%sthread pinning enabled",buf);
				} else{
					sprintf(buf2,"%sthread pinning disabled",buf);
				}
   infostruct->codesequence = bi_strdup(buf2);
   infostruct->xaxistext = bi_strdup("Used Memory in kByte");
   infostruct->num_measurements = nMeasurements;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = omp_get_max_threads();
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 1;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = functionCount;
   infostruct->base_xaxis = 10;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
