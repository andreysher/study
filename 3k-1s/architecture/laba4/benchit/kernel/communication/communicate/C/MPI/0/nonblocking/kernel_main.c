/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/communicate/C/MPI/0/nonblocking/kernel_main.c $
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
 *     1. MPI_Isend/Irecv   ->  rank 0 send to rank 1,...,n
 *     2. MPI_Issend/Irecv  ->  rank 0 send to rank 1,...,n
 *     3. MPI_Irsend/Irecv  ->  rank 0 send to rank 1,...,n
 *     4. MPI_Ibsend/Irecv  ->  rank 0 send to rank 1,...,n
 */
/* attention: in principle the measurement of Ibsend should only show the memory bandwidth on rank 0
 *            and NOT the communication bandwidth (it's is measured here only because of completeness) */

int numfct = 4;
int numberSends[4];

float *buffer;

/* dtime: the time for a single measurement in seconds */
double *dtime;

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   char *p = 0;
   mydata_t * penv;
   int i, j;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

   MPI_Comm_rank(MPI_COMM_WORLD, &(penv->commrank));
   MPI_Comm_size(MPI_COMM_WORLD, &(penv->commsize));

   infostruct->codesequence
      = bi_strdup("start kernel; make different types auf nonblocking point-to-point communication;");
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
   infostruct->numfunctions = numfct * 3 * 2; // functions * 3 values (min/max/mean) * 2 (time/bandwith);

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   i = 0;
   numberSends[(i/3)/2] = penv->commsize - 1;
   for (j=i; j<i+3; j++)
      infostruct->yaxistexts[j] = bi_strdup("s");
   for (j=i; j<i+3; j++)
      infostruct->selected_result[j] = SELECT_RESULT_LOWEST;
   for (j=i; j<i+3; j++)
      infostruct->base_yaxis[j] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Isend (min time)");
   infostruct->legendtexts[i+1] = bi_strdup("Isend (max time)");
   infostruct->legendtexts[i+2] = bi_strdup("Isend (mean time)");
   i+=3;
   for (j=i; j<i+3; j++)
      infostruct->yaxistexts[j] = bi_strdup("Byte/s");
   for (j=i; j<i+3; j++)
      infostruct->selected_result[j] = SELECT_RESULT_HIGHEST;
   for (j=i; j<i+3; j++)
      infostruct->base_yaxis[j] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Isend (min bw)");
   infostruct->legendtexts[i+1] = bi_strdup("Isend (max bw)");
   infostruct->legendtexts[i+2] = bi_strdup("Isend (mean bw)");
   i+=3;

   numberSends[(i/3)/2] = penv->commsize - 1;
   for (j=i; j<i+3; j++)
      infostruct->yaxistexts[j] = bi_strdup("s");
   for (j=i; j<i+3; j++)
      infostruct->selected_result[j] = SELECT_RESULT_LOWEST;
   for (j=i; j<i+3; j++)
      infostruct->base_yaxis[j] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Issend (min time)");
   infostruct->legendtexts[i+1] = bi_strdup("Issend (max time)");
   infostruct->legendtexts[i+2] = bi_strdup("Issend (mean time)");
   i+=3;
   for (j=i; j<i+3; j++)
      infostruct->yaxistexts[j] = bi_strdup("Byte/s");
   for (j=i; j<i+3; j++)
      infostruct->selected_result[j] = SELECT_RESULT_HIGHEST;
   for (j=i; j<i+3; j++)
      infostruct->base_yaxis[j] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Issend (min bw)");
   infostruct->legendtexts[i+1] = bi_strdup("Issend (max bw)");
   infostruct->legendtexts[i+2] = bi_strdup("Issend (mean bw)");
   i+=3;

   numberSends[(i/3)/2] = penv->commsize - 1;
   for (j=i; j<i+3; j++)
      infostruct->yaxistexts[j] = bi_strdup("s");
   for (j=i; j<i+3; j++)
      infostruct->selected_result[j] = SELECT_RESULT_LOWEST;
   for (j=i; j<i+3; j++)
      infostruct->base_yaxis[j] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Irsend (min time)");
   infostruct->legendtexts[i+1] = bi_strdup("Irsend (max time)");
   infostruct->legendtexts[i+2] = bi_strdup("Irsend (mean time)");
   i+=3;
   for (j=i; j<i+3; j++)
      infostruct->yaxistexts[j] = bi_strdup("Byte/s");
   for (j=i; j<i+3; j++)
      infostruct->selected_result[j] = SELECT_RESULT_HIGHEST;
   for (j=i; j<i+3; j++)
      infostruct->base_yaxis[j] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Irsend (min bw)");
   infostruct->legendtexts[i+1] = bi_strdup("Irsend (max bw)");
   infostruct->legendtexts[i+2] = bi_strdup("Irsend (mean bw)");
   i+=3;

   numberSends[(i/3)/2] = penv->commsize - 1;
   for (j=i; j<i+3; j++)
      infostruct->yaxistexts[j] = bi_strdup("s");
   for (j=i; j<i+3; j++)
      infostruct->selected_result[j] = SELECT_RESULT_LOWEST;
   for (j=i; j<i+3; j++)
      infostruct->base_yaxis[j] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Ibsend (min time)");
   infostruct->legendtexts[i+1] = bi_strdup("Ibsend (max time)");
   infostruct->legendtexts[i+2] = bi_strdup("Ibsend (mean time)");
   i+=3;
   for (j=i; j<i+3; j++)
      infostruct->yaxistexts[j] = bi_strdup("Byte/s");
   for (j=i; j<i+3; j++)
      infostruct->selected_result[j] = SELECT_RESULT_HIGHEST;
   for (j=i; j<i+3; j++)
      infostruct->base_yaxis[j] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("Ibsend (min bw)");
   infostruct->legendtexts[i+1] = bi_strdup("Ibsend (max bw)");
   infostruct->legendtexts[i+2] = bi_strdup("Ibsend (mean bw)");
   i+=3;

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

   buffer = (float*)malloc(pmydata->maxsize * sizeof(float));

   bi_random_init(0, (unsigned long long)-1);
   for (ii=0; ii<pmydata->maxsize; ii++) {
      buffer[ii] = (float)bi_random32();
   }

   dtime = (double*)calloc(numfct*3, sizeof(double));

   /* alloc intern mpi buffer (buffer has space for each Bsend (size-1 times)
    * and additional space (like backup, if mpi lib have empty space between the partial buffers))*/
   MPI_Buffer_attach(malloc(pmydata->maxsize*4*pmydata->commsize),
      pmydata->maxsize*4*pmydata->commsize);

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
      communicate(pmydata->commrank, pmydata->commsize, i, buffer,
         imyproblemSize, &(dtime[i*3]));
   }

   if (pmydata->commrank == 0) {
      for (i=0; i<numfct*3; i++) {
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
         dresults[i*3*2+1] = dtime[i*3]; // min time
         dresults[i*3*2+2] = dtime[i*3+1]; // max time
         dresults[i*3*2+3] 
            = (dtime[i*3+2]!=INVALID_MEASUREMENT) ? dtime[i*3+2] 
               / numberSends[i] : INVALID_MEASUREMENT; // mean time

         byte = (double)imyproblemSize * 4;
         dresults[i*3*2+4]
            = (dtime[i*3+1]!=INVALID_MEASUREMENT) ? byte
               / dtime[i*3+1] : INVALID_MEASUREMENT; // min bw = bw / tmax
         dresults[i*3*2+5] = (dtime[i*3]!=INVALID_MEASUREMENT) ? byte
            / dtime[i*3] : INVALID_MEASUREMENT; // max bw = bw / tmin

         byte *= numberSends[i];
         dresults[i*3*2+6]
            = (dtime[i*3+2]!=INVALID_MEASUREMENT) ? byte
               / dtime[i*3+2] : INVALID_MEASUREMENT; // mean bw
      }
   }

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv) {
   mydata_t * pmydata = (mydata_t*)mdpv;
   char * buff;
   int buffsize=0;

   /* detach buffer from mpi lib, get pointer to the allocated buffer to free this buffer */
   MPI_Buffer_detach(&buff, &buffsize);
   if (buff)
      free(buff);

   if (buffer)
      free(buffer);
   if (dtime)
      free(dtime);

   if (pmydata)
      free(pmydata);
   return;
}


