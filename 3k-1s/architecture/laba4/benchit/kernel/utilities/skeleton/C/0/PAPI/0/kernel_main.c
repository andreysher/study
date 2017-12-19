/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/utilities/skeleton/C/0/PAPI/0/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: c PAPI kernel skeleton
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"

#include <papi.h>
#include <papiStdEventDefs.h>
/*  Header for local functions
 */
#include "work.h"

/** These variables will help us to keep the overview over the arrays
  * we access for our functions/data.
  */
/* Number of different ways an algorithm will be measured.
   Example: loop orders: ijk, ikj, jki, jik, kij, kji -> functionCount=6 with
   each different loop order in an own function. */
int functionCount;
/* Number of fixed functions we have per measurement.
   Example: execution time and MFLOPS are measured for each loop order
   -> valuesPerFunction=2 */
int valuesPerFunction;

int MIN, MAX, INCREMENT;

// this will handle the PAPI-Events

int papi_eventset = PAPI_NULL;
int papi_returnvalue = PAPI_OK;

// example event 1 fp ops
int event1 = PAPI_FP_OPS;
// example event 2 total L2 cache misses
int event2 = PAPI_L2_TCM;


/*  Header for local functions
 */
void evaluate_environment(void);
void handlePapiError(int);

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   int i = 0, j = 0; /* loop var for functionCount */
   /* get environment variables for the kernel */
   evaluate_environment();
   infostruct->codesequence = bi_strdup("work_[1|2]()");
   infostruct->xaxistext = bi_strdup("Problem Size");
   infostruct->num_measurements = (MAX-MIN+1)/INCREMENT;
   if((MAX-MIN+1) % INCREMENT != 0)
     infostruct->num_measurements++;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;

   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   /* B ########################################################*/
   functionCount = 2; /* number versions of this algorithm (ijk, ikj, kij, ... = 6 */
   valuesPerFunction = 3; /* time measurement FP_OPS, INT_OPS */
   /*########################################################*/
   infostruct->numfunctions = functionCount * valuesPerFunction;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   for (j = 0; j < functionCount; j++)
   {
      /* B ########################################################*/
      int index1 = 0 * functionCount + j;
      int index2 = 1 * functionCount + j;
      int index3 = 2 * functionCount + j;
      // 1st function
      infostruct->yaxistexts[index1] = bi_strdup("s");
      infostruct->selected_result[index1] = SELECT_RESULT_LOWEST;
      infostruct->base_yaxis[index1] = 0;
      // 2nd function
      infostruct->yaxistexts[index2] = bi_strdup("Floating Point Operations");
      infostruct->selected_result[index2] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[index2] = 0;
      /*########################################################*/
      // 3rd function
      infostruct->yaxistexts[index3] = bi_strdup("L2 Cache Misses");
      infostruct->selected_result[index3] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[index3] = 0;
      switch (j)
      {
         /* B ########################################################*/
         case 1: // 2nd version legend text; maybe (ikj)
            infostruct->legendtexts[index1] =
               bi_strdup("Calculation Time in s (2)"); // "... (ikj)"
            infostruct->legendtexts[index2] =
               bi_strdup("Floating Point Operations (2)"); // "... (ikj)"
            infostruct->legendtexts[index3] =
               bi_strdup("L2 Cache Misses (2)"); // "... (ijk)"
            break;
         case 0: // 1st version legend text; maybe (ijk)
         default:
            infostruct->legendtexts[index1] =
               bi_strdup("Calculation Time in s (1)"); // "... (ijk)"
            infostruct->legendtexts[index2] =
               bi_strdup("Floating Point Operations (1)"); // "... (ijk)"
            infostruct->legendtexts[index3] =
               bi_strdup("L2 Cache Misses (1)"); // "... (ijk)"
         /*########################################################*/
      }
   }
   if (DEBUGLEVEL > 3)
   {
      /* the next for loop: */
      /* this is for your information only and can be ereased if the kernel works fine */
      for (i = 0; i < infostruct->numfunctions; i++)
      {
         printf("yaxis[%2d]=%s\t\t\tlegend[%2d]=%s\n",
            i, infostruct->yaxistexts[i], i, infostruct->legendtexts[i]);
      }
   }
}

/** Implementation of the bi_init of the BenchIT interface.
 *  Here you have the chance to allocate the memory you need.
 *  It is also possible to allocate the memory at the beginning
 *  of every single measurement and to free the memory thereafter.
 *  But always making use of the same memory is faster.
 *  HAVE A LOOK INTO THE HOWTO !
 */
void* bi_init(int problemSizemax)
{
   mydata_t* mdp;
   const PAPI_hw_info_t *hwinfo = NULL;
   mdp = (mydata_t*)malloc(sizeof(mydata_t));
   if (mdp == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }
/*   if (problemSizemax > STEPS)
   {
      fprintf(stderr, "Illegal maximum problem size\n"); fflush(stderr);
      exit(127);
   }*/
   /* B ########################################################*/
   /* malloc our own arrays in here */
   /*########################################################*/
   
   IDL(3,printf("\nPAPI init..."));
   if((papi_returnvalue = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
   {
      IDL(3,printf("error\n"));
      exit(1);
   }
   IDL(3,printf("done\n"));
   hwinfo = PAPI_get_hardware_info();
   if (hwinfo != NULL)
   {
   IDL(3,printf("\n-------------- PAPI Infos ------------\n"));
   IDL(3,printf("Nodes in the system    :%i\n",hwinfo->nnodes));
   IDL(3,printf("CPUs per node          :%i\n",hwinfo->ncpu));
   IDL(3,printf("Total CPUs             :%i\n",hwinfo->totalcpus));
   IDL(3,printf("Vendor ID Number of CPU:%i\n",hwinfo->vendor));
   IDL(3,printf("Vendor ID String of CPU:%i\n",hwinfo->vendor_string));
   IDL(3,printf("Model Number of CPU    :%i\n",hwinfo->model));
   IDL(3,printf("Model String of CPU    :%i\n",hwinfo->model_string));
   IDL(3,printf("Revision Number of CPU :%i\n",hwinfo->vendor_string));
   IDL(3,printf("(estimated) MHz of CPU :%f\n",hwinfo->mhz));
   IDL(3,printf("\n--------------------------------------\n",hwinfo->nnodes));
   }
   IDL(2,printf("This system has %d available counters.\n",PAPI_num_counters()));
   IDL(3,printf("\nPAPI create eventset..."));
   papi_returnvalue = PAPI_create_eventset(&papi_eventset);
   if (papi_returnvalue!=PAPI_OK)
   {
     IDL(3,printf("error\n"));
     handlePapiError(papi_returnvalue);
   }
   IDL(3,printf("done\n"));
   IDL(3,printf("\nPAPI add eventset... (1)"));
   papi_returnvalue = PAPI_add_event(papi_eventset,event1);
   if (papi_returnvalue!=PAPI_OK)
   {
     IDL(3,printf("error\n"));
     handlePapiError(papi_returnvalue);
   }
   IDL(3,printf("done\n"));
   IDL(3,printf("\nPAPI add eventset... (2)"));
   papi_returnvalue = PAPI_add_event(papi_eventset,event2);
   if (papi_returnvalue!=PAPI_OK)
   {
     IDL(3,printf("error\n"));
     handlePapiError(papi_returnvalue);
   }
   IDL(3,printf("done\n"));
   return (void*)mdp;
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
 */
int bi_entry(void* mdpv, int problemSize, double* results)
{
  /* timeInSecs: the time for a single measurement in seconds */
  long long  eventValues[2]={0,0};
  double timeInSecs = 0.0;
  /* flops stores the calculated FLOPS */
  // double flops = 0.0;
  /* j is used for loop iterations */
  int j = 0;
  /* cast void* pointer */
  mydata_t* mdp = (mydata_t*)mdpv;

  /* calculate real problemSize */
  problemSize = MIN + (problemSize - 1) * INCREMENT;

  /* check wether the pointer to store the results in is valid or not */
  if (results == NULL) return 1;

  /* B ########################################################*/
  /* maybe some init stuff in here */
  mdp->dummy = 0;
  /*########################################################*/

  for (j = 0; j < functionCount; j++)
  {
    /* B ########################################################*/
    int index1 = 0 * functionCount + j;
    int index2 = 1 * functionCount + j;
    int index3 = 2 * functionCount + j;
    /* choose version of algorithm */
    switch (j) {
      case 1: // 2nd version legend text; maybe (ikj)
        /* take start time, do measurement, and take end time */
        bi_startTimer();
        papi_returnvalue = PAPI_start(papi_eventset);
        if (papi_returnvalue!=PAPI_OK)
        {
          handlePapiError(papi_returnvalue);
        }
        work_2(problemSize);
        papi_returnvalue = PAPI_stop(papi_eventset,eventValues);
        if (papi_returnvalue!=PAPI_OK)
        {
          handlePapiError(papi_returnvalue);
        }
        timeInSecs = bi_stopTimer() ;
        break;
      case 0: // 1st version legend text; maybe (ijk)
      default:
        /* take start time, do measurement, and take end time */
        bi_startTimer();
        papi_returnvalue = PAPI_start(papi_eventset);
        if (papi_returnvalue!=PAPI_OK)
        {
          handlePapiError(papi_returnvalue);
        }
        work_1(problemSize);
        papi_returnvalue = PAPI_stop(papi_eventset,eventValues);
        if (papi_returnvalue!=PAPI_OK)
        {
          handlePapiError(papi_returnvalue);
        }
        timeInSecs = bi_stopTimer() ;
    }
    /* store the results in results[1], results[2], ...
    * [1] for the first function, [2] for the second function
    * and so on ...
    * the index 0 always keeps the value for the x axis
    */
    /* B ########################################################*/
    // the xaxis value needs to be stored only once!
    if (j == 0) results[0] = (double)problemSize;
    results[index1 + 1] = timeInSecs;
    results[index2 + 1] = eventValues[0];
    results[index3 + 1] = eventValues[0];
    /*########################################################*/
  }

  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   papi_returnvalue = PAPI_cleanup_eventset(papi_eventset);
   if (papi_returnvalue!=PAPI_OK)
   {
     handlePapiError(papi_returnvalue);
   }
   /* Free all memory and data structures, EventSet must be empty. */
   papi_returnvalue = PAPI_destroy_eventset(&papi_eventset);
   if (papi_returnvalue!=PAPI_OK)
   {
     handlePapiError(papi_returnvalue);
   }
   mydata_t* mdp = (mydata_t*)mdpv;
   /* B ########################################################*/
   /* may be freeing our arrays here */
   /*########################################################*/
   if (mdp) free(mdp);
   return;
}
/********************************************************************/
/*************** End of interface implementations *******************/
/********************************************************************/

/** Tries to measure the timer overhead for a single call to PAPI_get_real_usec().
 *  @return the calculated overhead in seconds
 */
double gettimeroverhead()
{
  long long start, stop;
  int s;

  start = PAPI_get_real_usec();
  for (s = 0; s < 10000; s++)
  {
    PAPI_get_real_usec();
  }
  stop = PAPI_get_real_usec();
  // E10 : 4 because of loop 6 because of usec instead of sec
  return (stop - start) / 1E10;
}

/* Reads the environment variables used by this kernel. */
void evaluate_environment()
{
   int errors = 0;
   char * p = 0;
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMSIZE_MIN", 0);
   if (p == 0) errors++;
   else MIN = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMSIZE_MAX", 0);
   if (p == 0) errors++;
   else MAX = atoi(p);
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMSIZE_INCREMENT", 0);
   if (p == 0) errors++;
   else INCREMENT = atoi(p);
   if (errors > 0)
   {
      fprintf(stderr, "There's at least one environment variable not set!\n");
      fprintf(stderr, "This kernel needs the following environment variables:\n");
      fprintf(stderr, "BENCHIT_KERNEL_PROBLEMSIZE_MIN\n");
      fprintf(stderr, "BENCHIT_KERNEL_PROBLEMSIZE_MAX\n");
      fprintf(stderr, "BENCHIT_KERNEL_PROBLEMSIZE_INCREMENT\n");
      fprintf(stderr, "\nThis kernel will iterate from BENCHIT_KERNEL_PROBLEMSIZE_MIN\n\
to BENCHIT_KERNEL_PROBLEMSIZE_MAX, incrementing by\n\
BENCHIT_KERNEL_PROBLEMSIZE_INCREMENT with each step.\n");
      exit(1);
   }
}
void handlePapiError(int papiReturn)
{
    if (papiReturn==PAPI_OK)
      return;
    fprintf(stderr, "PAPI error %d: %s\n",papiReturn,PAPI_strerror(papiReturn));
    exit(1);
}
