/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
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
int functionCount = 5; //add,sub,mul,div,mad
/* Number of fixed functions we have per measurement.
   Example: execution time and MFLOPS are measured for each loop order
   -> valuesPerFunction=2 */
int valuesPerFunction = 2; // Latency + Throughput

int maxThreads;
int maxThreadsPerBlock;
size_t timesSize;
size_t outSize;
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
   infostruct->codesequence = bi_strdup("");
   
   cudaDeviceProp props;
   cudaGetDeviceProperties(&props, 0);
   char CC[20];
   sprintf(CC, "CC %g", props.major + (float)props.minor/10);
   
   infostruct->kerneldescription = bi_strdup(CC);
   infostruct->xaxistext = bi_strdup("#Threads");
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
   for (j = 0; j < functionCount; j++){
      /* B ########################################################*/
      int index1 = 0 * functionCount + j;
      int index2 = 1 * functionCount + j;
      // 1st function
      infostruct->yaxistexts[index1] = bi_strdup("clocks");
      infostruct->selected_result[index1] = SELECT_RESULT_LOWEST;
      infostruct->base_yaxis[index1] = 0;
      // 2nd function
      infostruct->yaxistexts[index2] = bi_strdup("ops/clock/SM");
      infostruct->selected_result[index2] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[index2] = 0;
      switch (j){
         case 0: // 1. version legend text
         default:
            infostruct->legendtexts[index1] = bi_strdup("Latency (add)");
            infostruct->legendtexts[index2] = bi_strdup("Throughput (add)");
            break;
         case 1: // 2. version legend text
            infostruct->legendtexts[index1] = bi_strdup("Latency (sub)");
            infostruct->legendtexts[index2] = bi_strdup("Throughput (sub)");
            break;
         case 2: // 3. version legend text
            infostruct->legendtexts[index1] = bi_strdup("Latency (mul)");
            infostruct->legendtexts[index2] = bi_strdup("Throughput (mul)");
            break;
         case 3: // 4. version legend text
            infostruct->legendtexts[index1] = bi_strdup("Latency (div)");
            infostruct->legendtexts[index2] = bi_strdup("Throughput (div)");
            break;
         case 4: // 5. version legend text
            infostruct->legendtexts[index1] = bi_strdup("Latency (mad)");
            infostruct->legendtexts[index2] = bi_strdup("Throughput (mad)");
            break;
      }
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
 
struct SMData{
 	uint Min, Max, MinStop, Diff;
 	SMData(): Min((uint)-1), MinStop((uint)-1), Max(0), Diff(0){}
};
typedef std::map<uint,SMData> SM_MinMaxMap;
 
 #define WARPSIZE 32
 #define OPs (128*2)
int bi_entry(void* mdpv, int problemSize, double* results)
{
	int i, j;
  /* cast void* pointer */
  mydata_t* mdp = (mydata_t*)mdpv;

  /* calculate real problemSize */
  problemSize = bi_get_list_element(problemSize); 
  
  // the xaxis value needs to be stored only once!
  results[0] = (double)problemSize;
  
  int dimBlock = min(problemSize, mdp->maxThreadsPerBlock);
  int dimGrid = (problemSize + dimBlock - 1)/dimBlock;
	for(i=0; i < functionCount;i++){
	  switch(i){
	  	case 0:
	  		CUDA_CHECK_KERNEL_SYNC(EXEC_FUNC(ADD)<<<dimGrid, dimBlock>>>(problemSize, mdp->d_ts, mdp->d_out, 4, 6, 2));
	  		break;
	  	case 1:
	  		CUDA_CHECK_KERNEL_SYNC(EXEC_FUNC(SUB)<<<dimGrid, dimBlock>>>(problemSize, mdp->d_ts, mdp->d_out, 4, 6, 2));
	  		break;
	  	case 2:
	  		CUDA_CHECK_KERNEL_SYNC(EXEC_FUNC(MUL)<<<dimGrid, dimBlock>>>(problemSize, mdp->d_ts, mdp->d_out, 4, 6, 2));
	  		break;
	  	case 3:
	  		CUDA_CHECK_KERNEL_SYNC(EXEC_FUNC(DIV)<<<dimGrid, dimBlock>>>(problemSize, mdp->d_ts, mdp->d_out, 4, 6, 2));
	  		break;
	  	case 4:
	  		CUDA_CHECK_KERNEL_SYNC(EXEC_FUNC(MAD)<<<dimGrid, dimBlock>>>(problemSize, mdp->d_ts, mdp->d_out, 4, 6, 2));
	  		break;
	  }
  	CUDA_CHECK(cudaMemcpy(mdp->h_ts, mdp->d_ts, timesSize, cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(mdp->h_out, mdp->d_out, outSize, cudaMemcpyDeviceToHost));
  	SM_MinMaxMap smMap;
  	uint minDiff = (uint)-1;
  	uint diff;
  	uint sumTime = 0;
  	for(j=0; j < (problemSize+WARPSIZE-1)/WARPSIZE;++j){
  		int sm = mdp->h_out[j+1];
  		SMData mm = smMap[sm];
  		diff = mdp->h_ts[j*2+1] - mdp->h_ts[j*2];
  		minDiff = min(minDiff, diff);
  		//mm.Diff += diff;
  		mm.Min = min(mm.Min, mdp->h_ts[j*2]);
  		mm.MinStop = min(mm.MinStop, mdp->h_ts[j*2+1]);
  		mm.Max = max(mm.Max, mdp->h_ts[j*2+1]);
  		smMap[sm] = mm;
  		//if(mdp->h_out[j/2+1]==1) printf("%u - %u (%u)| %u - %u(%u)\n",mm.Min, mm.Max, mm.Max-mm.Min, mdp->h_ts[j], mdp->h_ts[j+1], mdp->h_ts[j+1] - mdp->h_ts[j]);
  	}
  	if(smMap.size()>13){
  		printf("\nSMs: %i\n",smMap.size());
  		printf("Threads %i",problemSize);
  		uint lastSM = mdp->h_out[1];
  		uint lastT=0;
  		for(j=1; j < problemSize; j++){
  			if(mdp->h_out[j+1]!=lastSM){
  				printf(" t%i-%i=%i(%i-%i=%i):%i",lastT,j-1,j-lastT,lastT/32,(j-1)/32,(j-1)/32-lastT/32+1,lastSM);
  				lastT=j;
  				lastSM=mdp->h_out[j+1];
  			}
  		}
  		printf(" t%i-%i=%i(%i-%i=%i):%i",lastT,j-1,j-lastT,lastT/32,(j-1)/32,(j-1)/32-lastT/32+1,lastSM);
  		fflush(stdout);
  	} 
  	//printf("\nThreads: %i SMs: %i\n", problemSize, smMap.size());
  	//double maxOPS=0, ops;
  	bool error = false;
  	for(SM_MinMaxMap::const_iterator it = smMap.begin(); it != smMap.end(); it++){
	  	sumTime+= it->second.Max - it->second.Min;
	  	//sumTime += it->second.Diff;
	  	//printf("SM %i: %u %i\n",it->first, it->second.Max - it->second.Min, it->second.Ops);
	  	//ops = (double)it->second.Ops/sumTime;
	  	//maxOPS=max(maxOPS,ops);
	  	if(it->first >1000 || it->second.MinStop<it->second.Min){ //Check for reschedule or overflow
	  		printf("\nINVALID SM (%i) or overflow detected! Threads: %i Func %i\n", it->first, problemSize, i);
	  		error = true;
	  		break;
	  	}
	  	//printf("%i SM%i: %i-%i=%i\n",problemSize, it->first, it->second.Min, it->second.Max, it->second.Max - it->second.Min);
  	}
  	results[i+1]=(error) ? INVALID_MEASUREMENT: ((double)minDiff) / OPs;//Latency
  	results[functionCount+i+1]=(error) ? INVALID_MEASUREMENT: ((double)OPs*problemSize)/sumTime;//Throughput
  	//results[functionCount+i+1]=maxOPS;//Throughput
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
   if (mdp == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }
   
   maxThreads = (int)bi_get_list_maxelement();
   timesSize = maxThreads * 2 * sizeof(uint);
   outSize = (maxThreads+1) * sizeof(uint);
   
   mdp->h_ts = (uint*) malloc(timesSize);
   mdp->h_out = (uint*) malloc(outSize);
   CUDA_CHECK(cudaMalloc((void**)&(mdp->d_ts),timesSize));
   CUDA_CHECK(cudaMalloc((void**)&(mdp->d_out),outSize));
   
   cudaDeviceProp props;
   CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
   mdp->maxThreadsPerBlock = min(props.maxThreadsPerBlock, maxThreadsPerBlock);
   return (void*)mdp;
}

// Clean up the memory
void bi_cleanup(void* mdpv)
{
   mydata_t* mdp = (mydata_t*)mdpv;
   free(mdp->h_ts);
   free(mdp->h_out);
   CUDA_CHECK(cudaFree(mdp->d_ts));
   CUDA_CHECK(cudaFree(mdp->d_out));
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

   p = bi_getenv("BENCHIT_KERNEL_NUMTHREADS", 0);
   if(p==0) errors++;
   else bi_parselist(p);

   p = bi_getenv("BENCHIT_KERNEL_THREADS_PER_BLOCK", 0);
   if (p == 0) errors++;
   else maxThreadsPerBlock = atoi(p);

   if (errors > 0)
   {
      fprintf(stderr, "There's at least one environment variable not set!\n");
      fprintf(stderr, "This kernel needs the following environment variables:\n");
      fprintf(stderr, "BENCHIT_KERNEL_NUMTHREADS\n");
      fprintf(stderr, "BENCHIT_KERNEL_THREADS_PER_BLOCK\n");
      exit(1);
   }
}
