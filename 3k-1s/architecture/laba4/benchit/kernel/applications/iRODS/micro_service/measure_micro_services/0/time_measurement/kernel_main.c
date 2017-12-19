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
  
   p = bi_getenv("BENCHIT_KERNEL_NUMBER_RUNS", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->number_runs = atoi(p);
   
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
 *
 *  In this case it also executes all given measurements to get the 
 *  name and the number of the measured Micro-Services for the plot.
 */
void bi_getinfo(bi_info * infostruct)
{
   mydata_t * penv;
   
   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   evaluate_environment(penv);
   
   infostruct->codesequence = bi_strdup("start kernel; do nothing; ");
   infostruct->kerneldescription = bi_strdup("iRods (Micro-Service): Time measurement of user defined Micro-Services");
   infostruct->xaxistext = bi_strdup("Run Number");
   infostruct->num_measurements = penv->number_runs;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   
   /* Starts the measurement */
   if (system(penv->path_script) != 0)
   {
   	fprintf(stderr,"Error: Couldn't start the script %s.", penv->path_script);   	
		exit(1);
	 }	
	penv->CSV = fopen(penv->path_temp, "r");
   if(NULL == penv->CSV) 
   {
      fprintf(stderr, "Error: Can't open the result file\n");
      exit(1);
   }
   /* Determines the number of measured Micro-Services */
   double time;
   myinttype number_func = 0;
   char text1[10],text2[FUNC_NAME];
   while(fscanf(penv->CSV,"%s\t%s\t%lf\n",text1,text2,&time) != EOF)
   {
   	number_func++;
   }
   if(number_func == 0)
   {
   	fprintf(stderr,"\nError: Empty result file\n");
   	exit(1);
   }
   infostruct->numfunctions = number_func - 1;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
