/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
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
int functionCount = 1; //Latency
/* Number of fixed functions we have per measurement.
   Example: execution time and MFLOPS are measured for each loop order
   -> valuesPerFunction=2 */
int valuesPerFunction = 6;
size_t clocksSize;
int maxThreadsPerBlock;


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
	 int j;
   /* get environment variables for the kernel */
   evaluate_environment();
   infostruct->codesequence = bi_strdup("clock()");
   infostruct->xaxistext = bi_strdup("Threads");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;

   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;

   infostruct->numfunctions = functionCount * valuesPerFunction;

   cudaDeviceProp props;
   cudaGetDeviceProperties(&props, 0);
   maxThreadsPerBlock = props.maxThreadsPerBlock;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   int curCt=maxThreadsPerBlock;
   for (j = 0; j < valuesPerFunction; j++){
      infostruct->yaxistexts[j] = bi_strdup("Count");
      infostruct->selected_result[j] = SELECT_RESULT_AVERAGE;
      infostruct->base_yaxis[j] = 0;
      char txt[30];
      sprintf(txt, "Diff. clocks (%i Threads/Block)", curCt);
      curCt/=2;
      infostruct->legendtexts[j] = bi_strdup(txt);
   }
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
	int i, j, k;
  /* cast void* pointer */
  mydata_t* mdp = (mydata_t*)mdpv;

  /* calculate real problemSize */
  problemSize = bi_get_list_element(problemSize); 
  
  // the xaxis value needs to be stored only once!
  results[0] = (double)problemSize;
  
  int curCt=maxThreadsPerBlock;
	for(i=0; i < valuesPerFunction;i++){
  	int dimBlock = min(problemSize, curCt);
  	int dimGrid = (problemSize + dimBlock - 1)/dimBlock;
  	CUDA_CHECK_KERNEL_SYNC(getClock<<<dimGrid, dimBlock>>>(problemSize, mdp->d_clock));
  	CUDA_CHECK(cudaMemcpy(mdp->h_clock,mdp->d_clock,clocksSize,cudaMemcpyDeviceToHost));
  	memset(mdp->clocks,0,problemSize*sizeof(uint));
  	int ct=0;
  	for(j=0; j<problemSize; j++){
  		for(k=0; k<ct; k++){
	  	  if(mdp->h_clock[j]==mdp->clocks[k])
	  	    break;
	  	}
	  	if(k==ct){
  			mdp->clocks[k] = mdp->h_clock[j];
   		  ct++;
	  	}
    }
  	results[i+1]=ct;
  	curCt/=2;
	}
  return 0;
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
   if (mdp == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }
   
   int maxThreads = (int)bi_get_list_maxelement();
   clocksSize = maxThreads * sizeof(uint);
   
   mdp->h_clock = (uint*) malloc(clocksSize);
   mdp->clocks = (uint*) malloc(clocksSize);
   CUDA_CHECK(cudaMalloc((void**)&(mdp->d_clock),clocksSize));
   
   return (void*)mdp;
}

// Clean up the memory
void bi_cleanup(void* mdpv)
{
   mydata_t* mdp = (mydata_t*)mdpv;
   free(mdp->h_clock);
   free(mdp->clocks);
   CUDA_CHECK(cudaFree(mdp->d_clock));
   free(mdp);
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

   p = bi_getenv("BENCHIT_KERNEL_THREDADS", 0);
   if(p==0) errors++;
   else bi_parselist(p);

   if (errors > 0)
   {
      fprintf(stderr, "There's at least one environment variable not set!\n");
      fprintf(stderr, "This kernel needs the following environment variables:\n");
      fprintf(stderr, "BENCHIT_KERNEL_THREDADS\n");
      exit(1);
   }
}
