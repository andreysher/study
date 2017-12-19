/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/bandwidth/C/MPI/0/half2halfpingpong/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: pairwise Send/Recv between two MPI-Prozesses>
 *         this file holds all the functions needed by the 
 *         benchit-interface
 *******************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "pingpong.h"
#include <errno.h>

/* Reads the environment variables used by this kernel. */
void evaluate_environment(mydata_t * pmydata) {
   unsigned int errors = 0, inumentries = 0, ii = 0, ij = 0, found = 0;
   char *p = 0;
   char *s = 0;

   p = bi_getenv("BENCHIT_KERNEL_REPETITIONS", 0);
   if (p == NULL)
      errors++;
   else
      pmydata->repeat = strtol(p, (char **)NULL, 10);

   p = bi_getenv("BENCHIT_KERNEL_SHOW_PAIR_BANDWITH", 0);
   if (p == NULL)
      errors++;
   else
      pmydata->pair_bandwith = strtol(p, (char **)NULL, 10);

   p = bi_getenv("BENCHIT_KERNEL_SHOW_TOTAL_BANDWITH", 0);
   if (p == NULL)
      errors++;
   else
      pmydata->total_bandwith = strtol(p, (char **)NULL, 10);

   if (pmydata->pair_bandwith == 0 && pmydata->total_bandwith == 0) {
      fprintf(stderr, "nothing to be shown\n");
      fprintf(stderr,
              "neither BENCHIT_KERNEL_SHOW_PAIR_BANDWITH nor BENCHIT_KERNEL_SHOW_TOTAL_BANDWITH set\n");
      exit(1);
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &(pmydata->commrank));
   MPI_Comm_size(MPI_COMM_WORLD, &(pmydata->commsize));

   /* think positive */
   pmydata->empty_list = 'f';

   p = bi_getenv("BENCHIT_KERNEL_SENDERLIST", 0);
   s = bi_getenv("BENCHIT_KERNEL_RECEIVERLIST", 0);

   if ((p == NULL) || (s == NULL)) {
      errors++;
      fprintf(stderr,
              "BENCHIT_KERNEL_SENDERLIST or BENCHIT_KERNEL_RECEIVERLIST not set\n");
      fflush(stderr);
      pmydata->empty_list = 't';
   } else {
      if (0 == strlen(p) || 0 == strlen(s)) {
         fprintf(stderr,
                 "empty BENCHIT_KERNEL_SENDERLIST or BENCHIT_KERNEL_RECEIVERLIST -> using default sender-receiverlist\n");
         fflush(stderr);
         pmydata->empty_list = 't';
      } else {
         /* for senderlist */
         inumentries = 1;              /* first commata means 1 entry already
                                        * found */
         /* find out how many values are given in the list */
         while (p) {
            p = strstr(p, ",");
            if (p) {
               p++;
               inumentries++;
            }
         }
         if (inumentries != (pmydata->commsize / 2)) {  /* wrong list */
            fprintf(stderr,
                    "Listentries dont match MPI_Comm_size -> using default sender-receiverlist\n");
            fflush(stderr);
            pmydata->empty_list = 't';
         }
         /* allocate aray according to number of entries */
         pmydata->senderlist =
            (unsigned int *)malloc(sizeof(unsigned int) * inumentries);
         p = bi_getenv("BENCHIT_KERNEL_SENDERLIST", 0);
         /* entry bevore first commata */
         pmydata->senderlist[0] = strtol(p, (char **)NULL, 10);
         for (ii = 1; ii < inumentries; ii++) {
            p = strstr(p, ",") + 1;    /* pointer to next number in string */
            pmydata->senderlist[ii] = strtol(p, (char **)NULL, 10);
         }

         /* for receiverlist */
         inumentries = 1;              /* first commata means 1 entry already
                                        * found */
         /* find out how many values are given in the list */
         while (s) {
            s = strstr(s, ",");
            if (s) {
               s++;
               inumentries++;
            }
         }
         if (inumentries != (pmydata->commsize / 2)) {  /* wrong list */
            fprintf(stderr,
                    "Listentries dont match MPI_Comm_size - using default sender-receiverlist\n");
            fflush(stderr);
            pmydata->empty_list = 't';
         }
         /* allocate aray according to number of entries */
         pmydata->receiverlist =
            (unsigned int *)malloc(sizeof(unsigned int) * inumentries);
         s = bi_getenv("BENCHIT_KERNEL_RECEIVERLIST", 0);
         /* entry bevore first commata */
         pmydata->receiverlist[0] = strtol(s, (char **)NULL, 10);
         for (ii = 1; ii < inumentries; ii++) {
            s = strstr(s, ",") + 1;    /* pointer to next number in string */
            pmydata->receiverlist[ii] = strtol(s, (char **)NULL, 10);
         }

         if (pmydata->empty_list == 'f') {
            for (ii = 0; ii < pmydata->commsize - 1; ii++) {
               found = 0;
               for (ij = 0; ij < inumentries; ij++) {
                  if ((pmydata->senderlist[ij] == ii) || (pmydata->receiverlist[ij] == ii))
                     found = 1;
               }
               if (found == 0) {
                  fprintf(stderr,
                          "Mismatch in MPI_Comm_rank numbers -> using default sender-receiverlist\n");
                  fflush(stderr);
                  pmydata->empty_list = 't';
               }
            }
         }
      }
   }

   if (errors > 0) {
      fprintf(stderr, "There's at least one environment variable not set!\n");
      exit(1);
   }
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   char *p = 0;
   mydata_t *penv;
   unsigned int ii = 0;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence =
      bi_strdup
      ("each process sender xor receiver;  do MPI_send-MPI_recv simultaneous");
   infostruct->kerneldescription =
      bi_strdup
      ("kernel performs a MPI_send-MPI_recv with all MPI processes simultaneously");
   infostruct->xaxistext = bi_strdup("Messagesize in bytes");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 1;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;

   if (penv->pair_bandwith == 1)
      ii++;
   if (penv->total_bandwith == 1)
      ii++;
   infostruct->numfunctions = ii;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   if (penv->pair_bandwith == 1 && penv->total_bandwith == 0) {
      infostruct->yaxistexts[0] = bi_strdup("bytes/s");
      infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[0] = 0;
      infostruct->legendtexts[0] =
         bi_strdup("bandwidth of one communication-pair in bytes/s");
   }

   if (penv->pair_bandwith == 0 && penv->total_bandwith == 1) {
      infostruct->yaxistexts[0] = bi_strdup("bytes/s");
      infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[0] = 0;
      infostruct->legendtexts[0] =
         bi_strdup("total bandwidth of all communication-pairs in bytes/s");
   }

   if (penv->pair_bandwith == 1 && penv->total_bandwith == 1) {
      infostruct->yaxistexts[0] = bi_strdup("bytes/s");
      infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[0] = 0;
      infostruct->legendtexts[0] =
         bi_strdup("bandwidth of one communication-pair in bytes/s");
      infostruct->yaxistexts[1] = bi_strdup("bytes/s");
      infostruct->selected_result[1] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[1] = 0;
      infostruct->legendtexts[1] =
         bi_strdup("total bandwidth of all communication-pairs in bytes/s");
   }

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
   unsigned int ii, maxsize;

   pmydata = (mydata_t *) malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);

   maxsize = (unsigned int) bi_get_list_maxelement();
   pmydata->maxsize = maxsize;

   /* allocate and fill msg-msg_string for mpi-communication */
   pmydata->msg_string = (unsigned int *)malloc(pmydata->maxsize);
   if (pmydata->msg_string == 0) {
      fprintf(stderr, "Allocation of structure pmydata->msg_string failed\n");
      fflush(stderr);
      exit(127);
   }
   for (ii = 0; ii < pmydata->maxsize / sizeof(unsigned int); ii++) {
      pmydata->msg_string[ii] = (unsigned int)rand();
   }

   IDL(3,
       printf("\nrank=%d msg_stringcontent:%d\n", pmydata->commrank,
              pmydata->commsize));
   for (ii = 0; ii < pmydata->maxsize; ii++) {
      IDL(3, printf("%d ", pmydata->msg_string[ii]));
   }
   IDL(3, printf("\n"));

   IDL(3, printf("\nrank=%d size=%d\n", pmydata->commrank, pmydata->commsize));
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
int bi_entry(void *mdpv, int iproblemSize, double *dresults) {
   /* dstart, dend: the start and end time of the measurement */
   /* dtime: the time for a single measurement in seconds */
   double dstart = 0.0, dend = 0.0, dtime = 0.0;
   /* flops stores the calculated FLOPS */
   double dres = 0.0;
   /* ii is used for loop iterations */
   unsigned int ii = 0, isender = 0, ireceiver = 0;
   unsigned long int imsgsize = 0;
   /* cast void* pointer */
   mydata_t *pmydata = (mydata_t *) mdpv;

   IDL(3, printf("\nrank=%d entered bi_entry\n", pmydata->commrank));

   /* there was no sender/receiverlist supplied */
   if (pmydata->empty_list == 't') {
      /* lower half is sender, upper half is receiver */
      if (pmydata->commrank < (pmydata->commsize / 2)) {
         isender = pmydata->commrank;
         ireceiver = pmydata->commrank + (pmydata->commsize / 2);
      } else {
         isender = pmydata->commrank - (pmydata->commsize / 2);
         ireceiver = pmydata->commrank;
      }
      if (isender > (pmydata->commsize - 1) ||
          ireceiver > (pmydata->commsize - 1)) {
         fprintf(stderr, "\nrank=%d sender=%d receiver=%d\n", pmydata->commrank,
                 isender, ireceiver);
         fflush(stderr);
         isender = 0;
         ireceiver = 1;
      }
   } else {                            /* we have those two lists */
      for (ii = 0; ii < pmydata->commsize / 2; ii++) {
         if ((pmydata->senderlist[ii] == pmydata->commrank) || (pmydata->receiverlist[ii] == pmydata->commrank)) {
            isender = pmydata->senderlist[ii];
            ireceiver = pmydata->receiverlist[ii];
         }
      }
      /*if (pmydata->commrank < (pmydata->commsize / 2)) {
         isender = pmydata->senderlist[pmydata->commrank];
         ireceiver = pmydata->receiverlist[pmydata->commrank];
      } else {
         isender =
            pmydata->senderlist[pmydata->commrank - (pmydata->commsize / 2)];
         ireceiver =
            pmydata->receiverlist[pmydata->commrank - (pmydata->commsize / 2)];
      }*/
   }
/* fprintf(stderr, "\nisender=%d, ireceiver=%d\n",isender, ireceiver);fflush(stderr);
 */

   imsgsize = (unsigned long int)bi_get_list_element(iproblemSize);

   /* check wether the pointer to store the results in is valid or not */
   if (pmydata->commrank == 0) {
      if (dresults == NULL) {
         fprintf(stderr, "\nrank=%d resultpointer not allocated - panic\n",
                 pmydata->commrank);
         fflush(stderr);
         return 1;
      }
   }

   /* get the actual time do the measurement / your algorythm get the actual
    * time */
   MPI_Barrier(MPI_COMM_WORLD);
   dstart = bi_gettime();
   pingpong(&isender, &ireceiver, pmydata, &imsgsize);
   MPI_Barrier(MPI_COMM_WORLD);
   dend = bi_gettime();

   IDL(3,
       printf("rank=%d Problemsize=%d, Value=%f\n", pmydata->commrank,
              iproblemSize, dres));

   if (pmydata->commrank == 0) {
      /* calculate the used time and FLOPS */
      dtime = dend - dstart;
      dtime -= dTimerOverhead;

      /* If the operation was too fast to be measured by the timer function,
       * mark the result as invalid */
      if (dtime < dTimerGranularity)
         dtime = INVALID_MEASUREMENT;

      /* store the results in results[1], results[2], ... * [1] for the first
       * function, [2] for the second function * and so on ... * the index 0
       * always keeps the value for the x axis */
      dresults[0] = (double)imsgsize;

      /* setting up y axis texts and properties */
      if (pmydata->pair_bandwith == 1 && pmydata->total_bandwith == 0) {
         dresults[1] = (double)(imsgsize * pmydata->repeat * 2 / dtime);
      }
      if (pmydata->pair_bandwith == 0 && pmydata->total_bandwith == 1) {
         dresults[1] =
            (double)(imsgsize * pmydata->repeat * pmydata->commsize / dtime);
      }

      if (pmydata->pair_bandwith == 1 && pmydata->total_bandwith == 1) {
         dresults[1] = (double)(imsgsize * pmydata->repeat * 2 / dtime);
         dresults[2] =
            (double)(imsgsize * pmydata->repeat * pmydata->commsize / dtime);
      }
   }

   return 0;
}

/** Clean up the memory 
 */
void bi_cleanup(void *mdpv) {
   mydata_t *pmydata = (mydata_t *) mdpv;

   if (pmydata)
      free(pmydata);
   return;
}


