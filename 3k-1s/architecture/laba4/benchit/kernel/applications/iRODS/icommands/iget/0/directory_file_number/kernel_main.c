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

   p = bi_getenv("BENCHIT_KERNEL_FILES_INC_FUNC", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->files_func = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_FILES_MIN", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->files_min = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_FILES_MAX", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->files_max = atoi(p);
      
   /* Calculates the number of measurements for BENCHIT_KERNEL_FILES_INC_FUNC=0 */   
   if(pmydata->files_func == 0)
   {
      p = bi_getenv("BENCHIT_KERNEL_FILES_INC", 0);
      if (p == NULL) 
         errors++;
      else 
         pmydata->files_inc = atoi(p);
         
      if((pmydata->files_inc == 1) && (pmydata->files_min == 0))
         pmydata->steps = pmydata->files_max - pmydata->files_min;
      else
      {
         myinttype diff;	
         diff = pmydata->files_max - pmydata->files_min;
         myinttype test = diff / pmydata->files_inc;
         if((diff % pmydata->files_inc) != 0)
            pmydata->steps = test + 2;
         else
            pmydata->steps = test + 1;
      }
   }
   /* Calculates the number of measurements for BENCHIT_KERNEL_FILES_INC_FUNC > 0 */ 
   else
   {
      if(pmydata->files_min == 0)
      {
         fprintf(stderr,"\nError: BENCHIT_KERNEL_FILES_MIN == 0 is not allowed for BENCHIT_KERNEL_FILES_INC_FUNC>0\n");
         exit(1);
      }
      if(pmydata->files_max == 0)
      {
         fprintf(stderr,"\nError: BENCHIT_KERNEL_FILES_MAX == 0 is not allowed for BENCHIT_KERNEL_FILES_INC_FUNC>0\n");
         exit(1);
      }
      /* BENCHIT_KERNEL_FILES_INC_FUNC = 1 */
      if(pmydata->files_func == 1)
      {
         double test_value = 0.0000001;
         double exp_min = log(pmydata->files_min) / log(2);
         double exp_max = log(pmydata->files_max) / log(2);
         myinttype help = 1;
         if(fabs(exp_max - (double) ((myinttype) exp_max)) > test_value)
            help++; 
         pmydata->steps = (myinttype) exp_max - (myinttype) exp_min + help;
      }
      /* BENCHIT_KERNEL_FILES_INC_FUNC = 2 */
      else if(pmydata->files_func == 2)
      {
         double test_value = 0.0000001;
         double exp_min = log10(pmydata->files_min);
         double exp_max = log10(pmydata->files_max);
         myinttype help = 1;
         if(fabs(exp_max - (double) ((myinttype) exp_max)) > test_value)
            help++;  
         pmydata->steps = (myinttype) (exp_max - exp_min) + help;
      }
      else
      {
      	fprintf(stderr,"\nError: BENCHIT_KERNEL_FILES_INC_FUNC -> Undefined value\n");
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
   p = bi_getenv("BENCHIT_KERNEL_NUMBER_RUNS", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->number_runs = atoi(p);
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
  
   if (errors > 0)http://start.ubuntu.com/9.10/
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
   infostruct->kerneldescription = bi_strdup("iRods (iget): Transfer of a directory with a specific amount of files");
   infostruct->xaxistext = bi_strdup("Number of Files");
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
   infostruct->yaxistexts[0] = bi_strdup("s");
   sprintf(file_info,"%s/s",penv->file_unit);
   infostruct->yaxistexts[1] = bi_strdup(file_info);

   infostruct->selected_result[0] = SELECT_RESULT_LOWEST;
   infostruct->selected_result[1] = SELECT_RESULT_LOWEST;
   /* setting up x axis texts and properties */
   //infostruct->base_yaxis[0] = 0; //logarythmic axis 10^x
   sprintf(file_info,"<iget %s %s %s> [Filesize: %d %s]",penv->resource,penv->num_threads,penv->protocol,(penv->file_block_size*penv->file_block_number),penv->file_unit);
   infostruct->legendtexts[0] = bi_strdup(file_info);
   sprintf(file_info,"<iget %s %s %s> [Filesize: %d %s][Rate]",penv->resource,penv->num_threads,penv->protocol,(penv->file_block_size*penv->file_block_number),penv->file_unit);
   infostruct->legendtexts[1] = bi_strdup(file_info);
   
   if(penv->files_func != 0)
   	infostruct->base_xaxis = 10; //logarythmic axis 10^x
   	
   /* free all used space */
   if (penv) free(penv);
}



/** Implementation of the bi_init of the BenchIT interface.
 *  Here you have the chance to allocate the memory you need.
 *  It is also possible to allocate the memory at the beginning
 *  of every single measurement and to free the memory thereafter.
 *  But always making use of the same memory is faster.
 *  HAVE A LOOK INTO THE HOWTO !
 *
 *  In this case it also execute all given measurements.
 */
void* bi_init(int problemSizemax)
{
   mydata_t * pmydata;
   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);
   
   if((pmydata->files_min < 0) || (pmydata->files_min > pmydata->files_max))
   {
      fprintf(stderr,"Error: BENCHIT_KERNEL_FILES_MIN is not big enough or bigger than BENCHIT_KERNEL_FILES_MAX");
      exit(1);
   }
   
   /* Calculate the minimum and maximum exponent for the different logarithm. */
   int i,j;
   myinttype file_number, file_number_temp, min_exp, max_exp;
   if(pmydata->files_func == 1)
   {
      min_exp = log(pmydata->files_min) / log(2);
      max_exp = log(pmydata->files_max) / log(2); 
   }
   if(pmydata->files_func == 2)
   {
      min_exp = log10(pmydata->files_min);
      max_exp = log10(pmydata->files_max); 
   }
   /* Executes all different measurements */
   file_number_temp = 0;
   for(i = 0; i < pmydata->steps; i++)
   {
      /* Sets the new file number for the measurement, dependent on the defined 
         Parameter BENCHIT_KERNEL_FILES_INC_FUNC */ 
      file_number = 0;
      if(pmydata->files_func == 0)
      {
         if(pmydata->files_min == 0)
         {	
            if((i == 0) || (pmydata->files_inc==1))	
               file_number = pmydata->files_min + 1 + (i * pmydata->files_inc);
            else
               file_number = pmydata->files_min + (i * pmydata->files_inc);
         }
         else
            file_number = pmydata->files_min + (i * pmydata->files_inc);
      }
      if(pmydata->files_func == 1)
      {
         if(i == 0)
            file_number = pmydata->files_min;
         else
            file_number = pow(2,min_exp + i);
      }
      if(pmydata->files_func == 2)
      {
         if(i == 0)
            file_number = pmydata->files_min;
         else
            file_number = pow(10,min_exp + i);
      }
      if(pmydata->files_max < file_number)
      	file_number = pmydata->files_max;
      	
      char help[100];
      
      if(i == 0)
      	file_number_temp = 1;
     /* Executes all repetitions of one measurement */ 
	   for(j = 0; j < pmydata->number_runs; j++)
	   {
        	fprintf(stdout,"\n\t\t<<<< Files: %d , Run: %d >>>>\n", file_number, (j + 1));
        	fflush(stdout);
        	sprintf(help,"$BENCHIT_SPEZIAL_SCRIPT %d %d %d",file_number, file_number_temp, (j + 1));
        	/*Starts the script defined in $BENCHIT_SPEZIAL_SCRIPT*/
        	if (system(help) != 0)
         	fprintf(stderr,"Error: Couldn't start the script %s.", pmydata->path_script);
	   }
      file_number_temp = file_number;  
   }
   pmydata->CSV = fopen(pmydata->path_temp, "r");
   if(NULL == pmydata->CSV) 
   {
      fprintf(stderr, "Error: Can't open the result file\n");
      exit(1);
   }
   
   return (void *)pmydata;
}



/** The central function within each kernel. This function
 *  is called for each measurement step seperately.
 *  @param  mdpv         a pointer to the structure created in bi_init,
 *                       it is the pointer the bi_init returns
 *  @param  problemSize  the actual problemSize
 *  @param  results      a pointer to a field of doubles, the
 *                       size of the field depends on the number
 *                       of functions, there are #functions+1
 *                       doubles
 *  @return 0 if the measurement was sucessfull, something
 *          else in the case of an error
 *
 *  In this case no measurement will started. Only the results will collected.
 */
int bi_entry(void * mdpv, int iproblemSize, double * dresults)
{
   /* cast void* pointer */
   mydata_t * pmydata = (mydata_t *) mdpv;

   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL) return 1;
   
   int file;
	 double time_real, time_user, time_system;
	 /* Gets the results of the result file */
   if(fscanf(pmydata->CSV,"%lf;%lf;%lf\n",&time_real,&time_user,&time_system) != EOF)
   {	
      if(fscanf(pmydata->CSV,"%d\n",&file) != EOF)
      {
      	dresults[0] = (double) file; // file id
      	dresults[1] = time_real; // time
      	int file_size = pmydata->file_block_size * pmydata->file_block_number * file;	
      	if((file_size == 0) || (time_real == 0))
      	{
      		if(file_size == 0)
      			fprintf(stderr,"Error: Filesize = 0, can't calculate result \n");
      		else
      			fprintf(stderr,"Error: Time = 0, can't calculate result \n");
      		dresults[2] = 0;
      	}
     	  else
      		dresults[2] = file_size/time_real; // rate
   		}
   		else
   			fprintf(stderr,"Error: No entrie for file number found.");
   }
   else
   	fprintf(stderr,"Error: No more entries in the result file");

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   mydata_t * pmydata = (mydata_t*)mdpv;
   fclose(pmydata->CSV);
   if (system("rm $BENCHIT_SPEZIAL_RESULT") != 0)
   	fprintf(stderr,"Error: Couldn't delete %s",pmydata->path_temp);
   if (pmydata) free(pmydata);
   
   return;
}


/********************************************************************
 * Log-History
 *
 *******************************************************************/ 
