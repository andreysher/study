/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: cuda kernel skeleton
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
   /* get environment variables for the kernel */
   evaluate_environment();
   infostruct->codesequence = bi_strdup("work_[1|2]()");
   infostruct->xaxistext = bi_strdup("Problem Size");
   infostruct->num_measurements = (MAX-MIN+1)/INCREMENT;
   if((MAX-MIN+1) % INCREMENT != 0)
     infostruct->num_measurements++;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
//   infostruct->base_xaxis = 0;

   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   /* B ########################################################*/
   functionCount = 2; /* number versions of this algorithm (ijk, ikj, kij, ... = 6 */
   valuesPerFunction = 2; /* time measurement and FLOPS (calculated) */
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
      //infostruct->selected_result[index3] = 0;
      //infostruct->base_yaxis[index3] = 0;
      switch (j)
      {
         /* B ########################################################*/
         case 1: // 2nd version legend text; maybe (ikj)
            infostruct->legendtexts[index1] =
               bi_strdup("Calculation Time (CPU)"); // "... (ikj)"
            infostruct->legendtexts[index2] =
               bi_strdup("FLOPS (CPU)"); // "... (ikj)"
            break;
         case 0: // 1st version legend text; maybe (ijk)
         default:
            infostruct->legendtexts[index1] =
               bi_strdup("Calculation Time (GPU)"); // "... (ijk)"
            infostruct->legendtexts[index2] =
               bi_strdup("FLOPS (GPU)"); // "... (ijk)"
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
   problemSizemax = MIN + (problemSizemax - 1) * INCREMENT;
   mydata_t* mdp;
   mdp = (mydata_t*)malloc(sizeof(mydata_t));
   if (mdp == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }
   size_t arraySize=problemSizemax * sizeof(int);
   mdp->a_host = (int*) malloc(arraySize);
   mdp->b_host = (int*) malloc(arraySize);
   CUDA_CHECK(cudaMalloc((void**)&(mdp->a_device),arraySize));
   CUDA_CHECK(cudaMalloc((void**)&(mdp->b_device),arraySize));
   int i;
   for (i=0;i<problemSizemax;i++) mdp->a_host[i]=i;
   CUDA_CHECK(cudaMemcpy(mdp->a_device,mdp->a_host,arraySize,cudaMemcpyHostToDevice));
   /*########################################################*/
   /* malloc our own arrays in here */
   /*########################################################*/
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

  /*########################################################*/
  /* maybe some init stuff in here */
  /*########################################################*/
  
  // the xaxis value needs to be stored only once!
  results[0] = (double)problemSize;

  for (j = 0; j < functionCount; j++)
  {
    /* B ########################################################*/
    int index1 = 0 * functionCount + j;
    int index2 = 1 * functionCount + j;
    /* choose version of algorithm */
    switch (j) {
      case 1: // 2nd version legend text; maybe (ikj)
        /* take start time, do measurement, and take end time */
        bi_startTimer(); moveCPU(mdp->a_host, mdp->b_host, problemSize); timeInSecs=bi_stopTimer();
        break;
      case 0: // 1st version legend text; maybe (ijk)
      default:
        CUDA_CHECK(cudaMemcpy(mdp->a_device,mdp->a_host,problemSize*sizeof(int),cudaMemcpyHostToDevice));
        int blocksize=min(problemSize,256);
  			dim3 dimBlock(blocksize);
  			dim3 dimGrid(ceil(problemSize/(float)blocksize));
        /* take start time, do measurement, and take end time */
        bi_startTimer();
        CUDA_CHECK_KERNEL_SYNC(moveGPU<<<dimGrid,dimBlock>>>(mdp->a_device,mdp->b_device,problemSize));
        timeInSecs=bi_stopTimer();
        CUDA_CHECK(cudaMemcpy(mdp->b_host,mdp->b_device,problemSize*sizeof(int),cudaMemcpyDeviceToHost));
        //maybe check results here
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
    /*########################################################*/
  }

  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   mydata_t* mdp = (mydata_t*)mdpv;
   /*########################################################*/
   /* may be freeing our arrays here */
   /*########################################################*/
   free(mdp->a_host);
   free(mdp->b_host);
   CUDA_CHECK(cudaFree(mdp->a_device));
   CUDA_CHECK(cudaFree(mdp->b_device));   
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
