/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/utilities/skeleton/C/0/PCL/0/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: c pcl kernel skeleton
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"
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

#ifdef BENCHIT_USE_PCL
int *wish_list;
int wish_nevents;
int *doable_list;
int doable_nevents;
unsigned int mode;
PCL_DESCR_TYPE descr;
#endif

/*  Header for local functions
 */
void evaluate_environment(void);

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   int i = 0, j = 0; /* loop var for functionCount */
#ifdef BENCHIT_USE_PCL
   int status = -1;
   mode = PCL_MODE_USER;
   /* B ########################################################*/
   wish_nevents = 1;
   /*########################################################*/
   wish_list = (int *)malloc(wish_nevents * sizeof(int));
   if (wish_list == 0)
   {
      fprintf(stderr, "Allocation of wish_list failed.\n"); fflush(stderr);
      exit(127);
   }
   /* B ########################################################*/
   wish_list[0] = PCL_INSTR;
   /*########################################################*/
   if(PCLinit(&descr) != PCL_SUCCESS)
   {
      fprintf(stderr, "Unable to initialize PCL.\n"); fflush(stderr);
      exit(1);
   }

   if (DEBUGLEVEL > 0) {
      /* the next four lines are only for your information and can be ereased later */
      printf("\n*******************************************************************************\n");
      PCLstatus(descr);
      printf("*******************************************************************************\n");
      fflush(stdout);
   }

   status = PCLcheck(descr, wish_list, wish_nevents, &doable_list, &doable_nevents, mode);
   if (status != PCL_SUCCESS) {
      printf("None of your PCL Events can be counted on this machine!\n"); fflush(stdout);
      exit(127);
   }

   if (DEBUGLEVEL > 1) {
      /* the next two for loops: */
      /* this is for your information only and can be ereased if the kernel works fine */
      for (i=0; i<wish_nevents; i++) {
         int j = 0, hit = 0;
         printf("wish_list event %d (%s) is ", i, PCLeventname(wish_list[i]));
         for (j=0; j<doable_nevents; j++) {
            if (wish_list[i] == doable_list[j]) {
               hit=1;
               break;
            }
         }
         if (hit == 0) printf("not ");
         printf("measurable\n");
      }
      for (i=0; i<doable_nevents; i++) {
         printf("doable_list event %d (%s)\n", i, PCLeventname(doable_list[i]));
      }
   }

   /* just in case, although it's really not neccessary after PCLcheck */
   if (PCLquery(descr, doable_list, doable_nevents, mode) != PCL_SUCCESS){
      printf("PCL event(s) not measurable\n"); fflush(stdout);
      exit(1);
   }
#endif
   (void) memset (infostruct, 0, sizeof (bi_info));
   /* get environment variables for the kernel */
   evaluate_environment();
   /* B ########################################################*/
   infostruct->codesequence = bi_strdup("work_*()");
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
   functionCount = 2; /* number versions of this algorithm (ijk, ikj, kij, ... = 6 */
   valuesPerFunction = 2; /* time measurement and MFLOPS (calculated) */
   /*########################################################*/
#ifdef BENCHIT_USE_PCL
      infostruct->numfunctions = functionCount*(valuesPerFunction + doable_nevents);
#else
      infostruct->numfunctions = functionCount*valuesPerFunction;
#endif

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   for (j = 0; j < functionCount; j++)
   {
      /* B ########################################################*/
      int index1 = 0 * functionCount + j;
      int index2 = 1 * functionCount + j;
      //int index3 = 2 * functionCount + j;
      // 1st function
      infostruct->yaxistexts[index1] = bi_strdup("s");
      infostruct->selected_result[index1] = SELECT_RESULT_LOWEST;
      infostruct->base_yaxis[index1] = 0;
      // 2nd function
      infostruct->yaxistexts[index2] = bi_strdup("FLOPS");
      infostruct->selected_result[index2] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[index2] = 0;
      /*########################################################*/
      // 3rd function
      //infostruct->yaxistexts[index3] = bi_strdup("");
      //infostruct->selected_result[index3] = SELECT_RESULT_HIGHEST;
      //infostruct->base_yaxis[index3] = 0;
#ifdef BENCHIT_USE_PCL
      for (i = 0; i < doable_nevents; i++)
      {
         char buf[180];
         int index = functionCount * (valuesPerFunction + i) + j;
         if (doable_list[i] < PCL_MFLOPS)
            sprintf(buf, "%160s per second", PCLeventname(doable_list[i]));
         else
            sprintf(buf, "%160s", PCLeventname(doable_list[i]));
         infostruct->yaxistexts[index] = bi_strdup(buf);
         infostruct->selected_result[index] = SELECT_RESULT_LOWEST;
         infostruct->base_yaxis[index] = 0;
      }
#endif
      switch (j)
      {
         /* B ########################################################*/
         case 1: // 2nd version legend text; maybe (ikj)
            infostruct->legendtexts[index1] =
               bi_strdup("Calculation Time in s (2)"); // "... (ikj)"
            infostruct->legendtexts[index2] =
               bi_strdup("FLOPS (2)"); // "... (ikj)"
#ifdef BENCHIT_USE_PCL
            for (i = 0; i < doable_nevents; i++)
            {
               char buf[180];
               int index = functionCount * (valuesPerFunction + i) + j;
               if (doable_list[i] < PCL_MFLOPS)
                  sprintf(buf, "%160s per second (2)", PCLeventname(doable_list[i]));
               else
                  sprintf(buf, "%160s (2)", PCLeventname(doable_list[i]));
               infostruct->legendtexts[index] = bi_strdup(buf);
            }
#endif
            break;
         case 0: // 1st version legend text; maybe (ijk)
         default:
            infostruct->legendtexts[index1] =
               bi_strdup("Calculation Time in s (1)"); // "... (ijk)"
            infostruct->legendtexts[index2] =
               bi_strdup("FLOPS (1)"); // "... (ijk)"
#ifdef BENCHIT_USE_PCL
            for (i = 0; i < doable_nevents; i++)
            {
               char buf[180];
               int index = functionCount * (valuesPerFunction + i) + j;
               if (doable_list[i] < PCL_MFLOPS)
                  sprintf(buf, "%160s per second (1)", PCLeventname(doable_list[i]));
               else
                  sprintf(buf, "%160s (1)", PCLeventname(doable_list[i]));
               infostruct->legendtexts[index] = bi_strdup(buf);
            }
#endif
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
#ifdef BENCHIT_USE_PCL
   mdp->doable_list = doable_list;
   mdp->doable_nevents = doable_nevents;
   mdp->mode = mode;
   mdp->descr = descr;
   mdp->i_result = (PCL_CNT_TYPE*)malloc(doable_nevents * sizeof(PCL_CNT_TYPE));
   if (mdp->i_result == 0)
   {
      fprintf(stderr, "Allocation of i_result failed.\n"); fflush(stderr);
      exit(127);
   }
   mdp->fp_result = (PCL_FP_CNT_TYPE*)malloc(doable_nevents * sizeof(PCL_FP_CNT_TYPE));
   if (mdp->fp_result == 0)
   {
      fprintf(stderr, "Allocation of fp_result failed.\n"); fflush(stderr);
      exit(127);
   }
#endif
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
   double timeInSecs = 0.0;
   /* flops stores the calculated FLOPS */
   double flops = 0.0;
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
   // the xaxis value needs to be stored only once!
   results[0] = (double)problemSize;
   for (j = 0; j < functionCount; j++)
   {
      /* B ########################################################*/
      int index1 = 0 * functionCount + j;
      int index2 = 1 * functionCount + j;
      /* reset of reused values */
#ifdef BENCHIT_USE_PCL
      int i = 0;
      for (i = 0; i < mdp->doable_nevents; i++)
      {
         mdp->i_result[i] = 0;
         mdp->fp_result[i] = 0.0;
      }
#endif
      /* choose version of algorithm */
      switch (j) {
         case 1: // 2nd version legend text; maybe (ikj)
            /* take start time, do measurement, and take end time */
#ifdef BENCHIT_USE_PCL
            if (PCLstart(mdp->descr, mdp->doable_list,
               mdp->doable_nevents, mdp->mode) != PCL_SUCCESS)
            {
               fprintf(stderr, "Problem with starting PCL events.\n");
               bi_cleanup(mdpv); fflush(stderr);
               exit(1);
            }
#endif
            bi_startTimer(); work_2(problemSize); timeInSecs = bi_stopTimer();
#ifdef BENCHIT_USE_PCL
            if (PCLstop(mdp->descr, mdp->i_result,
               mdp->fp_result, mdp->doable_nevents) != PCL_SUCCESS)
            {
               fprintf(stderr, "Problem with stopping PCL events.\n");
               bi_cleanup(mdpv); fflush(stderr);
               exit(1);
            }
#endif
            break;
         case 0: // 1st version legend text; maybe (ijk)
         default:
            /* take start time, do measurement, and take end time */
#ifdef BENCHIT_USE_PCL
            if (PCLstart(mdp->descr, mdp->doable_list,
               mdp->doable_nevents, mdp->mode) != PCL_SUCCESS)
            {
               fprintf(stderr, "Problem with starting PCL events.\n");
               bi_cleanup(mdpv); fflush(stderr);
               exit(1);
            }
#endif
            bi_startTimer(); work_1(problemSize); timeInSecs = bi_stopTimer();
#ifdef BENCHIT_USE_PCL
            if (PCLstop(mdp->descr, mdp->i_result,
               mdp->fp_result, mdp->doable_nevents) != PCL_SUCCESS)
            {
               fprintf(stderr, "Problem with stopping PCL events.\n");
               bi_cleanup(mdpv); fflush(stderr);
               exit(1);
            }
#endif
      }
      /* calculate the used time and FLOPS */
      /* If the operation was too fast to be measured by the timer function,
       * mark the result as invalid */
      if(timeInSecs == INVALID_MEASUREMENT){
         flops = INVALID_MEASUREMENT;
      }else{
         // this flops value is a made up! this calulations should be replaced
         // by something right for the choosen algorithm
         flops = (double)problemSize;
      }
      /* store the results in results[1], results[2], ...
      * [1] for the first function, [2] for the second function
      * and so on ...
      * the index 0 always keeps the value for the x axis
      */
      /* B ########################################################*/
      results[index1 + 1] = timeInSecs;
      results[index2 + 1] = flops;
#ifdef BENCHIT_USE_PCL
      for (i = 0; i < mdp->doable_nevents; i++)
      {
         int index = functionCount * (valuesPerFunction + i) + j;
         if (mdp->doable_list[i] < PCL_MFLOPS)
            results[index + 1] = (double)mdp->i_result[i] / (1.0 * timeInSecs);
         else
            results[index + 1] = mdp->fp_result[i];
      }
#endif
      /*########################################################*/
   }

   return (0);
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   mydata_t* mdp = (mydata_t*)mdpv;
   /* B ########################################################*/
   /* may be freeing our arrays here */
   /*########################################################*/
#ifdef BENCHIT_USE_PCL
   free(mdp->i_result);
   free(mdp->fp_result);
   free(mdp->doable_list);
   PCLexit(mdp->descr);
#endif
   if (mdp) free(mdp);
   return;
}
/********************************************************************/
/*************** End of interface implementations *******************/
/********************************************************************/

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
