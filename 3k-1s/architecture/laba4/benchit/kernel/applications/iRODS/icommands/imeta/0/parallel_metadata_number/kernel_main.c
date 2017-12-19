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


/* Reads some environment variables used by this kernel and 
   calculates some values for the measurements */
int set_inc(mydata_t * pmydata)
{
   int errors = 0;
   char * p = 0;

   p = bi_getenv("BENCHIT_KERNEL_META_INC_FUNC", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->meta_func = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_META_MIN", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->meta_min = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_META_MAX", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->meta_max = atoi(p);
   
   /* Calculates the number of measurements for BENCHIT_KERNEL_FILES_INC_FUNC=0 */   
   if(pmydata->meta_func == 0)
   {
      p = bi_getenv("BENCHIT_KERNEL_META_INC", 0);
      if (p == NULL) 
         errors++;
      else 
         pmydata->meta_inc = atoi(p);
         
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
   }
   /* Calculates the number of measurements for BENCHIT_KERNEL_FILES_INC_FUNC > 0 */ 
   else
   {
      if(pmydata->meta_min == 0)
      {
         fprintf(stderr,"\nError: BENCHIT_KERNEL_META_MIN == 0 is not allowed for BENCHIT_KERNEL_META_INC_FUNC>0\n");
         exit(1);
      }
      if(pmydata->meta_max == 0)
      {
         fprintf(stderr,"\nError: BENCHIT_KERNEL_META_MAX == 0 is not allowed for BENCHIT_KERNEL_META_INC_FUNC>0\n");
         exit(1);
      }
      /* BENCHIT_KERNEL_FILES_INC_FUNC = 1 */
      if(pmydata->meta_func == 1)
      {
         double test_value = 0.0000001;
         double exp_min = log(pmydata->meta_min) / log(2);
         double exp_max = log(pmydata->meta_max) / log(2);
         myinttype help = 1;
         if(fabs(exp_max - (double) ((myinttype) exp_max)) > test_value)
            help++; 
         pmydata->steps = (myinttype) exp_max - (myinttype) exp_min + help;
      }
      /* BENCHIT_KERNEL_FILES_INC_FUNC = 2 */
      else if(pmydata->meta_func == 2)
      {
         double test_value = 0.0000001;
         double exp_min = log10(pmydata->meta_min);
         double exp_max = log10(pmydata->meta_max);
         myinttype help = 1;
         if(fabs(exp_max - (double) ((myinttype) exp_max)) > test_value)
            help++;  
         pmydata->steps = (myinttype) (exp_max - exp_min) + help;
      }
      else
      {
      	fprintf(stderr,"\nError: BENCHIT_KERNEL_META_INC_FUNC -> Undefined value\n");
      	exit(1);
      }
      
   }
   
   return errors;
}

/* Reads the environment variables used by this kernel. */
void evaluate_environment(mydata_t * pmydata)
{
   int errors = 0;
   char * p = 0;
   
   errors += set_inc(pmydata);
   p = bi_getenv("BENCHIT_KERNEL_META_ATTRIBUTE", 0);
   if (p == NULL) 
      errors++;
   else 
   {
      if(atoi(p) == 0)
      	strcpy(pmydata->meta_attr,"Different Values");
      else
      	strcpy(pmydata->meta_attr,"Different Attribute Names");
   }
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
 */
void bi_getinfo(bi_info * infostruct)
{
   mydata_t * penv;
   char file_info[100];
   
   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   evaluate_environment(penv);
   infostruct->codesequence = bi_strdup("start kernel; do nothing; ");
   infostruct->kerneldescription = bi_strdup("iRods (imeta): Parallel transfer for a specific amount of metadata");
   infostruct->xaxistext = bi_strdup("Number of Metadata");
   infostruct->num_measurements = penv->steps * penv->number_runs;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 1;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
