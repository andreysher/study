/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/bandwidth/C/MPI/0/binarybroadcast/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: bandwidth for a mpi broadcast implemented with a binary tree and send
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "interface.h"
#include "binarybroadcast.h"

char *buffer;
int numfct=1;

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
   int i;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

   infostruct->codesequence
      = bi_strdup("start kernel; make initial point-to-point communication on root and pass on the message over the binary tree;");
   infostruct->kerneldescription
      = bi_strdup("bandwidth for a mpi broadcast implemented with a binary tree and send");
   infostruct->xaxistext = bi_strdup("bytes in sendbuffer");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 1;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = numfct * 2; // time and bandwith for numfct functions;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   i = 0;
   infostruct->yaxistexts[i] = bi_strdup("s");
   infostruct->selected_result[i] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[i] = 10; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("broadcast (time)");
   i++;
   infostruct->yaxistexts[i] = bi_strdup("Byte/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("broadcast (bw)");

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

   /* allocate send buffer */
   buffer = (char*)malloc(pmydata->maxsize * sizeof(char));
   if (buffer == NULL) {
      printf("malloc (%d bytes) failed in bi_init()\n", pmydata->maxsize);
      free(pmydata);
      MPI_Finalize();
      exit(127);
   }

   /* fill send buffer */
   for (ii=0; ii<pmydata->maxsize; ii++) {
      buffer[ii] = (char) 12;
   }

   /* allocate array for time measurement */
   dtime = (double*)calloc(numfct, sizeof(double));

   return (void *)pmydata;
}

/** number of bytes send in the binary tree broadcast
 */
double count_(int *version, myinttype *size) {
   switch (*version) {
      default:
         return 2 * (double)(*size);
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
int bi_entry(void * mdpv, int iproblemSize, double * dresults) {
   int i;
   int v = 1;
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
      IDL(2, printf("start tree(%d)\n", pmydata->commrank));
      dtime[i] = bi_gettime();
      mpibinarybroadcast_(buffer, &imyproblemSize,
         &pmydata->commrank, &pmydata->commsize);
      dtime[i] = bi_gettime() - dtime[i] - dTimerOverhead;
      IDL(2, printf("tree done(%d)\n", pmydata->commrank));
   }

   if (pmydata->commrank == 0) {
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
         dresults[i+1] = dtime[i]; // time

         byte = count_(&v, &imyproblemSize);
         dresults[i+2] = (dtime[i]!=INVALID_MEASUREMENT) ? byte
            / dtime[i] : INVALID_MEASUREMENT; // bandwidth
      }
   }

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv) {
   mydata_t * pmydata = (mydata_t*)mdpv;

   if (buffer)
      free(buffer);
   if (dtime)
      free(dtime);
   if (pmydata)
      free(pmydata);
}


