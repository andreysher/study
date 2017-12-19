/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/bandwidth/C/MPI/0/roundtrip/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Measure MPI bandwith for a round trip send algorithm
 *******************************************************************/

#include "roundtrip.h"

/* Reads the environment variables used by this kernel. 
 * see interface.h for bi_getenv("name", exit_on_error)
 */
void evaluate_environment(mydata_t * pmydata) {
   char *p = 0;

   /* add additional parameters, except BENCHIT_KERNEL_PROBLEMLIST from the
    * parameters file BENCHIT_KERNEL_PROBLEMLIST will be done in bi_getinfo */
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   char *p = 0;
   mydata_t *penv;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence =
      bi_strdup("if(root){#"
                "   send message to right hand#"
                "   wait for message from the left side#"
                "} else {#"
                "   wait for message from the left side#"
                "   send message to the right side#"
                "}#");
   infostruct->kerneldescription =
      bi_strdup("Measure MPI bandwith for a round trip send algorithm");
   infostruct->xaxistext = bi_strdup("Message Size");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 1;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 1;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   infostruct->yaxistexts[0] = bi_strdup("byte/s");
   infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[0] = 0;           // logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("round trip");

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
void *bi_init(int problemSizemax) {
   mydata_t *pmydata;
   uint64_t m;

   pmydata = (mydata_t *) malloc(sizeof(mydata_t));
   if (pmydata == NULL) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }

   m = (uint64_t) bi_get_list_maxelement();
   pmydata->maxsize = m;

   MPI_Comm_rank(MPI_COMM_WORLD, &(pmydata->commrank));
   MPI_Comm_size(MPI_COMM_WORLD, &(pmydata->commsize));

   pmydata->sendbuf = (char *)malloc(m * sizeof(MPI_CHAR));
   if (pmydata->sendbuf == NULL) {
      printf("malloc (%ju bytes) failed in bi_init() (rank %d)\n", m, pmydata->commrank);
      free(pmydata);
      MPI_Finalize();
      exit(127);
   }

   pmydata->recvbuf = (char *)malloc(m * sizeof(MPI_CHAR));
   if (pmydata->recvbuf == NULL) {
      printf("malloc (%ju bytes) failed in bi_init() (rank %d)\n", m, pmydata->commrank);
      free(pmydata);
      MPI_Finalize();
      exit(127);
   }

   return (void *)pmydata;
}

/** stuff from old kernel
 */
void init_(void *mdpv, uint64_t * size) {
   mydata_t *pmydata = (mydata_t *) mdpv;

/* to suppress compile/link warnings thrown by memset */
   if (0 != *size)
   {
      memset(pmydata->sendbuf, (int)1, *size);
      memset(pmydata->recvbuf, (int)2, *size);
   }
}

/** stuff from old kernel
 */
double getmpientryoverhead(void *mdpv) {
   double start = 0.0, stop = 0.0;
   uint64_t nu = 0, ii = 0;

   init_(mdpv, &nu);
   start = bi_gettime();
   for (ii = 0; ii < 1000; ii++) {
      entry_(mdpv, &nu);
      /* MPI_Barrier(MPI_COMM_WORLD); */
   }
   stop = bi_gettime();
   return (stop - start) / 1000;
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
int bi_entry(void *mdpv, int iproblemSize, double *dresults) {
   /* dstart, dend: the start and end time of the measurement */
   /* dtime: the time for a single measurement in seconds */
   double dstart = 0.0, dend = 0.0, dtime = 0.0, bytes = 0.0;
   static double calloverhead = 0;
   /* i,j,k is used for loop iterations */
   myinttype ii = 0, numloop = 0;
   uint64_t imyproblemSize = 0;
   /* cast void* pointer */
   mydata_t *pmydata = (mydata_t *) mdpv;

   if (calloverhead == 0.0) {
      calloverhead = getmpientryoverhead(mdpv);
      if (calloverhead == 0.0)
         calloverhead = MINTIME;
   }

   /* get current problem size from problemlist */
   imyproblemSize = (uint64_t) bi_get_list_element(iproblemSize);

   /* initialize send / recv buffer */
   init_(mdpv, &imyproblemSize);

   /* check wether the pointer to store the results in is valid or not */
   if (pmydata->commrank == 0) {
      if (dresults == NULL) {
         fprintf(stderr, "\nrank=%d resultpointer not allocated - panic\n",
                 pmydata->commrank);
         fflush(stderr);
         return 1;
      }
   }

   IDL(2,
       printf("start round trip (rank %d, msgsize %ju)\n", pmydata->commrank,
              imyproblemSize));

   /* get the actual time do the measurement / your algorythm get the actual
    * time */
   numloop = 1;
   do {
      MPI_Barrier(MPI_COMM_WORLD);

      dstart = bi_gettime();
      for (ii = 0; ii < numloop; ii++)
         entry_(mdpv, &imyproblemSize);
      dend = bi_gettime();

      MPI_Bcast(&dstart, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&dend, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      numloop <<= 3;	/* numloop = numloop * 8 */
   } while (dend-dstart < 0.01);
   numloop >>= 3;	/* undo last shift */

   /* calculate the used time and #Operations */
   dtime = dend - dstart;
   dtime -= numloop * calloverhead;
   bytes = (double)imyproblemSize * (double)numloop * (double)sizeof(MPI_CHAR);

   if (pmydata->commrank == 0) {
      /* If the operation was too fast to be measured by the timer function,
       * mark the result as invalid */
      if (dtime < dTimerGranularity)
         dtime = INVALID_MEASUREMENT;

      /* store the results in results[1], results[2], ... [1] for the first
       * function, [2] for the second function and so on ... the index 0 always 
       * keeps the value for the x axis */
      dresults[0] = (double)imyproblemSize;
      dresults[1] =
         (dtime != INVALID_MEASUREMENT) ? bytes / dtime : INVALID_MEASUREMENT;
   }

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void *mdpv) {
   mydata_t *pmydata = (mydata_t *) mdpv;

   if (pmydata) {
      /* free send/recv buffer */
      if (pmydata->sendbuf)
         free(pmydata->sendbuf);
      if (pmydata->recvbuf)
         free(pmydata->recvbuf);

      free(pmydata);
   }
   return;
}
