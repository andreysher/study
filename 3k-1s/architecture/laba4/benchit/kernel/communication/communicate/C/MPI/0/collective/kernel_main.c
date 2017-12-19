/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/communicate/C/MPI/0/collective/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare the needed time for different MPI communication methodes
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "interface.h"
#include "communicate.h"

/*  Functions:
 *     1. MPI_Bcast       ->  rank 0 send to rank 1,...,n
 *     2. MPI_Scatter     ->  rank 0 send to rank 1,...,n
 *     3. MPI_Scatterv    ->  rank 0 send to rank 1,...,n
 *     4. MPI_Gather      ->  rank 1,...,n send to rank 0
 *     5. MPI_Gatherv     ->  rank 1,...,n send to rank 0
 *     6. MPI_Allgather   ->  rank i send to rank j (i,j=1,...,n; i!=j)
 *     7. MPI_Allgatherv  ->  rank i send to rank j (i,j=1,...,n; i!=j)
 *     8. MPI_Alltoall    ->  rank i send to rank j (i,j=1,...,n; i!=j)
 *     9. MPI_Alltoallv   ->  rank i send to rank j (i,j=1,...,n; i!=j)
 */

int numfct=9;
int numberSends[9];

float *sendbuffer;
float *recvbuffer;

/* dtime: the time for a single measurement in seconds */
double *dtime;
int *sendcounts, *displs;

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   char *p = 0;
   mydata_t * penv;
   int i;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

   MPI_Comm_rank(MPI_COMM_WORLD, &(penv->commrank));
   MPI_Comm_size(MPI_COMM_WORLD, &(penv->commsize));

   infostruct->codesequence
      = bi_strdup("start kernel; make different types auf collectiv communication;");
   infostruct->kerneldescription
      = bi_strdup("compare the needed time for different MPI communication methodes");
   infostruct->xaxistext
      = bi_strdup("elements in sendbuffer (# of floats)");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 1;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 2 * numfct;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   i = 0;
   numberSends[i/2] = penv->commsize - 1;
   infostruct->yaxistexts[i] = bi_strdup("s");
   infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[i] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Bcast (time)");
   i++;
   infostruct->yaxistexts[i] = bi_strdup("Byte/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Bcast (bw)");
   i++;

   numberSends[i/2] = penv->commsize - 1;
   infostruct->yaxistexts[i] = bi_strdup("s");
   infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[i] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Scatter (time)");
   i++;
   infostruct->yaxistexts[i] = bi_strdup("Byte/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Scatter (bw)");
   i++;

   numberSends[i/2] = penv->commsize - 1;
   infostruct->yaxistexts[i] = bi_strdup("s");
   infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[i] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Scatterv (time)");
   i++;
   infostruct->yaxistexts[i] = bi_strdup("Byte/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Scatterv (bw)");
   i++;

   numberSends[i/2] = penv->commsize - 1;
   infostruct->yaxistexts[i] = bi_strdup("s");
   infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[i] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Gather (time)");
   i++;
   infostruct->yaxistexts[i] = bi_strdup("Byte/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Gather (bw)");
   i++;

   numberSends[i/2] = penv->commsize - 1;
   infostruct->yaxistexts[i] = bi_strdup("s");
   infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[i] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Gatherv (time)");
   i++;
   infostruct->yaxistexts[i] = bi_strdup("Byte/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Gatherv (bw)");
   i++;

   numberSends[i/2] = penv->commsize * (penv->commsize - 1);
   infostruct->yaxistexts[i] = bi_strdup("s");
   infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[i] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Allgather (time)");
   i++;
   infostruct->yaxistexts[i] = bi_strdup("Byte/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Allgather (bw)");
   i++;

   numberSends[i/2] = penv->commsize * (penv->commsize - 1);
   infostruct->yaxistexts[i] = bi_strdup("s");
   infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[i] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Allgatherv (time)");
   i++;
   infostruct->yaxistexts[i] = bi_strdup("Byte/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Allgatherv (bw)");
   i++;

   numberSends[i/2] = penv->commsize * (penv->commsize - 1);
   infostruct->yaxistexts[i] = bi_strdup("s");
   infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[i] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Alltoall (time)");
   i++;
   infostruct->yaxistexts[i] = bi_strdup("Byte/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Alltoall (bw)");
   i++;

   numberSends[i/2] = penv->commsize * (penv->commsize - 1);
   infostruct->yaxistexts[i] = bi_strdup("s");
   infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[i] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Alltoallv (time)");
   i++;
   infostruct->yaxistexts[i] = bi_strdup("Byte/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Alltoallv (bw)");
   i++;

   /* free all used space */
   if (penv)
      free(penv);
}

/** Implementation of the bi_init of the BenchIT interface.
 *  Here you have the chance to allocate the memory you need.
 *  It is also possible to allocate the memory at the beginning
 *  of every single measurement and to free the memory thereafter.
 *  But always making use of the same memory is faster.
 *  HAVE A LOOK INTO THE HOWTO !
 */
void* bi_init(int problemSizemax) {
   mydata_t * pmydata;
   myinttype ii, maxsize;

   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }
   MPI_Comm_rank(MPI_COMM_WORLD, &(pmydata->commrank));
   MPI_Comm_size(MPI_COMM_WORLD, &(pmydata->commsize));

   maxsize = (myinttype) bi_get_list_maxelement();
   pmydata->maxsize = maxsize;

   IDL(3, printf("\nrank=%d size=%d\n", pmydata->commrank,
      pmydata->commsize));
   IDL(3, printf("max=%d\n", pmydata->maxsize));

   sendbuffer = (float*)malloc((pmydata->maxsize * pmydata->commsize)
      * sizeof(float));
   bi_random_init(0, (unsigned long long)-1);
   for (ii=0; ii<(pmydata->maxsize * pmydata->commsize); ii++) {
      sendbuffer[ii] = (float)bi_random32();
   }
   /* space for all recvs, largest for alltoall */
   recvbuffer = (float*)malloc((pmydata->maxsize * pmydata->commsize)
      * sizeof(float));

   sendcounts = (int*)malloc(pmydata->commsize * sizeof(int));
   displs = (int*)malloc(pmydata->commsize * sizeof(int));
   for (ii=0; ii<pmydata->commsize; ii++)
      displs[ii] = ii * pmydata->maxsize;

   dtime = (double*)malloc(numfct * sizeof(double));

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
 */
int bi_entry(void * mdpv, int iproblemSize, double * dresults) {
   int i;
   double byte;

   /* ii is used for loop iterations */
   myinttype ii = 0, imyproblemSize = 0;
   /* cast void* pointer */
   mydata_t * pmydata = (mydata_t *) mdpv;

   IDL(3, printf("\nrank=%d entered bi_entry\n", pmydata->commrank));
   /* calculate real problemSize */
   imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);

   for (i=0; i<pmydata->commsize; i++)
      sendcounts[i] = imyproblemSize;

   /* check wether the pointer to store the results in is valid or not */
   if (pmydata->commrank == 0) {
      if (dresults == NULL) {
         fprintf(stderr,
            "\nrank=%d resultpointer not allocated - panic\n",
            pmydata->commrank);
         fflush(stderr);
         return 1;
      }
   }

   /* get the actual time
    * do the measurement / your algorythm
    * get the actual time
    */
   for (i=0; i<numfct; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      communicate(pmydata->commrank, pmydata->commsize, i,
         sendbuffer, recvbuffer, imyproblemSize, sendcounts, displs,
         &(dtime[i]));
   }

   if (pmydata->commrank == 0) {
      /* calculate the used time and FLOPS */
      for (i=0; i<numfct; i++) {
         /* If the operation was too fast to be measured by the timer function,
          * mark the result as invalid 
          */
         if (dtime[i] < dTimerGranularity)
            dtime[i] = INVALID_MEASUREMENT;
      }

      /* store the results in results[1], results[2], ...
       * [1] for the first function, [2] for the second function
       * and so on ...
       * the index 0 always keeps the value for the x axis
       */
      dresults[0] = (double)imyproblemSize;
      for (i=0; i<numfct; i++) {
         dresults[i*2+1] = dtime[i];

         byte = numberSends[i] * (double)imyproblemSize * 4;
         dresults[i*2+2] = (dtime[i]!=INVALID_MEASUREMENT) ? byte
            / dtime[i] : INVALID_MEASUREMENT;
      }
   }

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv) {
   mydata_t * pmydata = (mydata_t*)mdpv;

   if (sendbuffer)
      free(sendbuffer);
   if (recvbuffer)
      free(recvbuffer);
   if (sendcounts)
      free(sendcounts);
   if (displs)
      free(displs);

   if (dtime)
      free(dtime);

   if (pmydata)
      free(pmydata);
   return;
}


