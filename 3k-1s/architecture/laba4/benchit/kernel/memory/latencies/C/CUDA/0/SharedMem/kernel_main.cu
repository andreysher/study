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
int functionCount = 3; //Latency
/* Number of fixed functions we have per measurement.
   Example: execution time and MFLOPS are measured for each loop order
   -> valuesPerFunction=2 */
int valuesPerFunction = 1;
int arrayLength;
int arraySize;
int iterations;

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
	 int index1;
   /* get environment variables for the kernel */
   evaluate_environment();
   infostruct->codesequence = bi_strdup("accessMem(cycles, count)");
   infostruct->xaxistext = bi_strdup("Accessed values");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;

   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;

   infostruct->numfunctions = functionCount * valuesPerFunction;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   for(index1 = 0; index1 < infostruct->numfunctions; index1++){
     infostruct->yaxistexts[index1] = bi_strdup("Cycles");
     infostruct->selected_result[index1] = SELECT_RESULT_LOWEST;
     infostruct->base_yaxis[index1] = 0;
     infostruct->legendtexts[index1] = bi_strdup("Latency");
   }
   infostruct->legendtexts[0] = bi_strdup("Same adress");
   infostruct->legendtexts[1] = bi_strdup("Same bank");
   infostruct->legendtexts[2] = bi_strdup("Succ. banks");
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
   
   cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
   cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
   
   mdp->h_array = (uint*) malloc(arraySize);
   //Add 1 dummy value
   CUDA_CHECK(cudaMalloc((void**)&(mdp->d_array),arraySize+sizeof(uint)));
   mdp->h_duration = (uint*) malloc(2*sizeof(uint));
   CUDA_CHECK(cudaMalloc((void**)&(mdp->d_duration),2*sizeof(uint)));

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
	int i, j;
  /* cast void* pointer */
  mydata_t* mdp = (mydata_t*)mdpv;

  /* calculate real problemSize */
  problemSize = bi_get_list_element(problemSize); 
  
  // the xaxis value needs to be stored only once!
  results[0] = (double)problemSize;
  
	for(i=0; i < functionCount;i++){
  	for(j = 0; j < arrayLength; j++)
  		mdp->h_array[j] = 0;
		int stride;
		if(i == 0){
  		stride=0;
  	}else if(i==1){
  		stride=1;
  	}else{
  		stride=2;
  	}
  	
  	CUDA_CHECK(cudaMemcpy(mdp->d_array,mdp->h_array,arraySize,cudaMemcpyHostToDevice));
  	
  	dim3 dimBlock(1);
  	dim3 dimGrid(1);
  	CUDA_CHECK_KERNEL_SYNC(testLatencyN<<<dimGrid, dimBlock, arraySize>>>(mdp->d_array, arrayLength, problemSize, stride, iterations, mdp->d_duration));
  	CUDA_CHECK(cudaMemcpy(mdp->h_duration,mdp->d_duration,2*sizeof(uint),cudaMemcpyDeviceToHost));
  	results[i+1]=mdp->h_duration[0] / (1 * iterations);
	}
  return 0;
}

// Clean up the memory
void bi_cleanup(void* mdpv)
{
   mydata_t* mdp = (mydata_t*)mdpv;
   free(mdp->h_array);
   free(mdp->h_duration);
   CUDA_CHECK(cudaFree(mdp->d_array));
   CUDA_CHECK(cudaFree(mdp->d_duration));
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

   p = bi_getenv("BENCHIT_KERNEL_ACCESSCOUNT", 0);
   if(p==0) errors++;
   else bi_parselist(p);

   p = bi_getenv("BENCHIT_KERNEL_ARRAYLENGTH", 0);
   if(p==0) errors++;
   else arrayLength = atoi(p);
   arraySize = arrayLength * sizeof(uint);

   p = bi_getenv("BENCHIT_KERNEL_ITERATIONS", 0);
   if(p==0) errors++;
   else iterations = atoi(p);

   if (errors > 0)
   {
      fprintf(stderr, "There's at least one environment variable not set!\n");
      fprintf(stderr, "This kernel needs the following environment variables:\n");
      fprintf(stderr, "BENCHIT_KERNEL_ACCESSCOUNT\n");
      fprintf(stderr, "BENCHIT_KERNEL_ARRAYSIZE\n");
      fprintf(stderr, "BENCHIT_KERNEL_ITERATIONS\n");
      exit(1);
   }
}
