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
   p = bi_getenv("BENCHIT_KERNEL_NUMBER_FILES", 0);
   if (p == NULL) 
   	errors++;
   else 
   	pmydata->files_number = atoi(p);
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
   p = bi_getenv("BENCHIT_RESULT_VIEW", 0);
   if (p == NULL) 
      errors++;
   else 
      pmydata->result_view = atoi(p);
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
   infostruct->kerneldescription = bi_strdup("iRods (iput): Detailed parallel transfer for a specific amount of files");
   infostruct->xaxistext = bi_strdup("File ID");
   infostruct->num_measurements = penv->number_runs * penv->files_number;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 5;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   infostruct->yaxistexts[0] = bi_strdup("s");
   infostruct->yaxistexts[1] = bi_strdup("s");
   infostruct->yaxistexts[2] = bi_strdup("s");
   infostruct->yaxistexts[3] = bi_strdup("s");
   if(penv->result_view == 0)
   	infostruct->yaxistexts[4] = bi_strdup("s");
   else
   {
    sprintf(file_info,"%s/s",penv->file_unit);
   	infostruct->yaxistexts[4] = bi_strdup(file_info);
   }
   infostruct->selected_result[0] = SELECT_RESULT_LOWEST;
   infostruct->selected_result[1] = SELECT_RESULT_LOWEST;
   infostruct->selected_result[2] = SELECT_RESULT_LOWEST;
   infostruct->selected_result[3] = SELECT_RESULT_LOWEST;
   infostruct->selected_result[4] = SELECT_RESULT_LOWEST;
       
   /* setting up x axis texts and properties */    
   //infostruct->base_yaxis[0] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("Environment");
   infostruct->legendtexts[1] = bi_strdup("Connection");
   infostruct->legendtexts[2] = bi_strdup("Logon");
   infostruct->legendtexts[3] = bi_strdup("Put File");
   if(penv->result_view == 0)
   	sprintf(file_info,"<iput %s %s %s> (Total Time[Filesize: %d %s])",penv->resource,penv->num_threads,penv->protocol,(penv->file_block_size*penv->file_block_number),penv->file_unit);
   else
    sprintf(file_info,"<iput %s %s %s> (Rate [Filesize: %d %s]) [second scale]",penv->resource,penv->num_threads,penv->protocol,(penv->file_block_size*penv->file_block_number),penv->file_unit);
	 infostruct->legendtexts[4] = bi_strdup(file_info);
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
   
   int i;
   char help[50];
   /* Execute all measurement runs */
   for(i = 0; i < pmydata->number_runs; i++)
	 {
   	fprintf(stdout,"\n\t\t<<<< Run: %d >>>>\n", (i + 1));
    fflush(stdout);
    sprintf(help,"$BENCHIT_SPEZIAL_SCRIPT %d", (i + 1));
    /*Starts the script defined in $BENCHIT_SPEZIAL_SCRIPT*/
    if (system(help) != 0)
   		fprintf(stderr,"Error: Couldn't start the script %s.", pmydata->path_script);
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
 	double time_all, time_env, time_conn, time_logon, time_getutil;
	/* Gets the results of the result file */
   if(fscanf(pmydata->CSV,"%d;%lf;%lf;%lf;%lf;%lf\n",&file,&time_all,&time_env,&time_conn,&time_logon,&time_getutil) != EOF)
   {	
      dresults[0] = (double) file; //file id
      dresults[1] = time_env; // time environment
      dresults[2] = time_conn; //time connection
      dresults[3] = time_logon; // time logon
      dresults[4] = time_getutil; // time get the file
      if(pmydata->result_view == 0)
      	dresults[5] = time_all; // total time (seconds)
      else
      {
      	int file_size = pmydata->file_block_size * pmydata->file_block_number;	
      	if((file_size == 0) || (time_all == 0))
      	{
      		if(file_size == 0)
      			fprintf(stderr,"Error: Filesize = 0, can't calculate result \n");
      		else
      			fprintf(stderr,"Error: Time = 0, can't calculate result \n");
      		dresults[5] = 0;
      	}
      	else
      		dresults[5] = file_size/time_all; // total time (rate)
   		}
   }
   else
   	fprintf(stderr,"Error: No more entries in th result file");

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
   //if (system("rm $BENCHIT_SPEZIAL_RESULT\"_extra\"") != 0)
   	//fprintf(stderr,"Error: Couldn't delete %s_extra",pmydata->path_temp);
   	
   if (pmydata) free(pmydata);
   
   return;
}


/********************************************************************
 * Log-History
 *
 *******************************************************************/ 
