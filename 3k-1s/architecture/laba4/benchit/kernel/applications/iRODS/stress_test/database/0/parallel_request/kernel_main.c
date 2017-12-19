/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 *
 * Kernel: 
 * 
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"

#include "data_struct.h"


/* Reads the environment variables used by this kernel. */
void evaluate_environment(mydata_t * pmydata)
{
   int errors = 0;
   char * p = 0;
   p = bi_getenv("BENCHIT_KERNEL_PROCESS_MIN", 0);
   if (p == NULL) 
   	errors++;
   else 
   	pmydata->meta_min = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_PROCESS_MAX", 0);
   if (p == NULL) 
   	errors++;
   else 
   	pmydata->meta_max = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_PROCESS_INC", 0);
   if (p == NULL) 
   	errors++;
   else 
   	pmydata->meta_inc = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_NUMBER_RUNS", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->number_runs = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_PROCESS_LOOP", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->threads_loop = atoi(p);
   p = bi_getenv("BENCHIT_SPEZIAL_SCRIPT", 0);
   if (p == NULL) 
      errors++;
   else 
   { 
     strncpy(pmydata->path_script,p,500);
   }
   p = bi_getenv("BENCHIT_SPEZIAL_RESULT", 0);
   if (p == NULL) 
      errors++;
   else 
   {
     strncpy(pmydata->path_temp,p,500);
   }
   
   if((pmydata->meta_inc == 1) && (pmydata->meta_min == 0))
   	pmydata->steps = pmydata->meta_max - pmydata->meta_min;
   else
   {
	 	myinttype diff;	
	 	diff = pmydata->meta_max - pmydata->meta_min;
   	myinttype test = diff / pmydata->meta_inc;
   	if((diff % pmydata->meta_inc) != 0)
   		pmydata->steps = test + 2;
   	else
   		pmydata->steps = test + 1;
	 }
  
   if (errors > 0)
   {
      fprintf(stderr, "There's at least one environment variable not set!\n");
      exit(1);
   }
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   mydata_t * penv;
   char file_info[100];
   
   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   evaluate_environment(penv);
   infostruct->codesequence = bi_strdup("start kernel; do nothing; ");
   infostruct->kerneldescription = bi_strdup("iRods: Parallel requests to test the behaviour of iRODS");
   infostruct->xaxistext = bi_strdup("Number of Processes");
   infostruct->num_measurements = penv->steps * penv->number_runs;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 2;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
