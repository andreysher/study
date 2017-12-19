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
   p = bi_getenv("BENCHIT_KERNEL_FILES_NUMBER", 0);
   if (p == NULL) 
   	errors++;
   else 
   	pmydata->file_number = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_FILES_START", 0);
   if (p == NULL) 
   	errors++;
   else 
   	pmydata->file_start = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_FILL_NUMBER", 0);
   if (p == NULL) 
   	errors++;
   else 
   	pmydata->fill_number = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_FILE_UNIT", 0);
   if (p == NULL)
      errors++;
   else 
   {
      if(strcmp(p,"") == 0)
      	strncpy(pmydata->file_unit,"Byte",5);
      else if((strcmp(p,"K") == 0) || (strcmp(p,"k") == 0))
      	strncpy(pmydata->file_unit,"KByte",5);
      else if((strcmp(p,"M") == 0) || (strcmp(p,"m") == 0))
      	strncpy(pmydata->file_unit,"MByte",5);
      else if((strcmp(p,"G") == 0) || (strcmp(p,"g") == 0))
      	strncpy(pmydata->file_unit,"GByte",5);
      else
      {
      	fprintf(stderr,"\nError: No valid file unit found");
      	errors++;
      }
   }
   p = bi_getenv("BENCHIT_KERNEL_FILE_BLOCK_SIZE", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->file_block_size = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_FILE_BLOCK_NUMBER", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->file_block_number = atoi(p);
   p = bi_getenv("BENCHIT_IRODS_RESC", 0);
   if (p == NULL) 
      errors++;
   else 
   {
      if(strcmp(p,"") == 0)
      	strncpy(pmydata->resource,"",1);
      else
      	sprintf(pmydata->resource,"-R %s",p);
   }
   p = bi_getenv("BENCHIT_IRODS_THREADS", 0);
   if (p == NULL) 
      errors++;
   else 
   {
      if(atoi(p) == -1)
      	strncpy(pmydata->resource,"",1);
      else
      	sprintf(pmydata->num_threads,"-N %s",p);
   }
   p = bi_getenv("BENCHIT_IRODS_PROT", 0);
   if (p == NULL) 
      errors++;
   else
   {
      if(atoi(p) == 0)
      	strncpy(pmydata->protocol,"",1);
      else
      	strncpy(pmydata->protocol,"-Q",3);
   }
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
   infostruct->kerneldescription = bi_strdup("iRods: Transfer of a directory with files to measure the used time by increasing overhead.");
   infostruct->xaxistext = bi_strdup("Total Number of Files");
   infostruct->num_measurements = penv->fill_number;
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
