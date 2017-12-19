/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/reduction/C/MPI/0/reduce/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare the flops for different MPI reduction methodes
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "interface.h"
#include "reduce.h"
#include <unistd.h>

/*  Functions:
 *     1. MPI_MIN
 *     2. MPI_MAX
 *     3. MPI_SUM
 *     4. MPI_PROD
 *     5. user defined vector addition (commutativ) with vectors in R^3
 *     6. user defined matmul (not commutativ, Winograd algorithm) with matricies in R^(2,2)
 */
/*  Example:
 *    rank0:    2  2  2  1 -1
 *    rank1:    1  2  3  4  1
 *    rank2:    4  3  2  1  0
 *    MPI_SUM=  7  7  7  6  0
 */
int numfct=6;

float *in, *out;

Vector *invec, *outvec;
Matrix *inmat, *outmat;

MPI_Datatype *myctypes;
MPI_Op *myops;

/* dtime: the time for a single measurement in seconds */
double *dtime;

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   char *p = 0;
   int i;


   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

   infostruct->codesequence
      = bi_strdup("start kernel; make different types auf reduction;");
   infostruct->kerneldescription
      = bi_strdup("compare the flops for different MPI reduction methodes");
   infostruct->xaxistext = bi_strdup("elements in sendbuffer");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 1;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = numfct;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   i = 0;
   infostruct->yaxistexts[i] = bi_strdup("flops/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("MIN");
   i++;

   infostruct->yaxistexts[i] = bi_strdup("flops/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("MAX");
   i++;

   infostruct->yaxistexts[i] = bi_strdup("flops/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("SUM");
   i++;

   infostruct->yaxistexts[i] = bi_strdup("flops/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("PROD");
   i++;

   infostruct->yaxistexts[i] = bi_strdup("flops/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("vecadd (commutativ)");
   i++;

   infostruct->yaxistexts[i] = bi_strdup("flops/s");
   infostruct->selected_result[i] = SELECT_RESULT_HIGHEST;
   infostruct->base_yaxis[i] = 0; //logarythmic axis 10^x
   infostruct->legendtexts[i] = bi_strdup("matmul (non-commutativ)");
   i++;
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

   /* in and out buffer for MPI-defined functions */
   in = (float*)malloc(pmydata->maxsize * sizeof(float));
   out = (float*)malloc(pmydata->maxsize * sizeof(float));

   /* in and out buffer for user defined functions */
   invec = (Vector*)malloc(pmydata->maxsize * sizeof(Vector));
   outvec = (Vector*)malloc(pmydata->maxsize * sizeof(Vector));
   inmat = (Matrix*)malloc(pmydata->maxsize * sizeof(Matrix));
   outmat = (Matrix*)malloc(pmydata->maxsize * sizeof(Matrix));

   if (pmydata->commrank % 2 == 0) {
      for (ii=0; ii<pmydata->maxsize; ii++)
         in[ii] = 1.0;
   } else {
      for (ii=0; ii<pmydata->maxsize; ii++)
         in[ii] = -1.0;
   }

   for (ii=0; ii<pmydata->maxsize; ii++) {
      invec[ii].x1 = -1.0;
      invec[ii].x2 = 1.0;
      invec[ii].x3 = 0.0;
   }

   if (pmydata->commrank % 2 == 0) {
      for (ii=0; ii<pmydata->maxsize; ii++) {
         inmat[ii].a11 = 2.0;
         inmat[ii].a12 = 0.0;
         inmat[ii].a21 = 0.0;
         inmat[ii].a22 = 2.0;
      }
   } else {
      for (ii=0; ii<pmydata->maxsize; ii++) {
         inmat[ii].a11 = 0.5;
         inmat[ii].a12 = 0.0;
         inmat[ii].a21 = 0.0;
         inmat[ii].a22 = 0.5;
      }
   }

   dtime = (double*)malloc(numfct * sizeof(double));

   myctypes = (MPI_Datatype*)malloc(numfct * sizeof(MPI_Datatype));
   myctypes[0] = MPI_FLOAT;
   myctypes[1] = MPI_FLOAT;
   myctypes[2] = MPI_FLOAT;
   myctypes[3] = MPI_FLOAT;
   MPI_Type_contiguous(3, MPI_FLOAT, &myctypes[4]);
   MPI_Type_commit(&myctypes[4]);
   MPI_Type_contiguous(4, MPI_FLOAT, &myctypes[5]);
   MPI_Type_commit(&myctypes[5]);

   myops = (MPI_Op*)malloc(numfct * sizeof(MPI_Op));
   myops[0] = MPI_MIN;
   myops[1] = MPI_MAX;
   myops[2] = MPI_SUM;
   myops[3] = MPI_PROD;
   MPI_Op_create(vecadd, 1, &myops[4]);
   MPI_Op_create(matmul, 0, &myops[5]);

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
   double flop;
   float res[4];
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
   /* MPI-defined */
   for (i=0; i<numfct-2; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      reduce(pmydata->commrank, pmydata->commsize, (void*)in,
         (void*)out, imyproblemSize, &myctypes[i], &myops[i],
         &dtime[i]);
      res[i] = out[0];
   }
   /* user-defined */
   MPI_Barrier(MPI_COMM_WORLD);
   i = 4;
   reduce(pmydata->commrank, pmydata->commsize, (void*)invec,
      (void*)outvec, imyproblemSize, &myctypes[i], &myops[i],
      &dtime[i]);
   MPI_Barrier(MPI_COMM_WORLD);
   i = 5;
   reduce(pmydata->commrank, pmydata->commsize, (void*)inmat,
      (void*)outmat, imyproblemSize, &myctypes[i], &myops[i],
      &dtime[i]);

   /* verify result */
   /* MPI_MIN -> -1, MPI_MAX -> 1, MPI_SUM -> 0, MPI_PROD -> 1|-1 */
   if (pmydata->commrank == 0) {
      if (res[0] != -1)
         fprintf(stderr,
            "\nincorrect result for MPI_MIN, expected -1, got %e",
            res[0]);
      fflush(stderr);
      if (res[1] != 1)
         fprintf(stderr,
            "\nincorrect result for MPI_MAX, expected 1, got %e",
            res[1]);
      fflush(stderr);
      if (res[2] != 0)
         fprintf(stderr,
            "\nincorrect result for MPI_SUM, expected 0, got %e",
            res[2]);
      fflush(stderr);
      if (res[3] != 1 && res[3] != -1)
         fprintf(
            stderr,
            "\nincorrect result for MPI_PROD, expected -1 or 1, got %e",
            res[3]);
      fflush(stderr);

      if (outvec[0].x1 != (double)-pmydata->commsize || outvec[0].x2
         != (double)pmydata->commsize || outvec[0].x3 != 0.0)
         fprintf(
            stderr,
            "\nincorrect result for vecadd, expected %e,%e,%e, got %e,%e,%e",
            -1*(double)pmydata->commsize, (double)pmydata->commsize,
            0.0, outvec[0].x1, outvec[0].x2, outvec[0].x3);
      fflush(stderr);
      if (outmat[0].a11 != 2.0 && outmat[0].a11 != 1.0)
         fprintf(
            stderr,
            "\nincorrect result for matmul, expected 1 or 2 for a11, got %e",
            outmat[0].a11);
      fflush(stderr);
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
      /* flops for MPI-defined */
      for (i=0; i<numfct-2; i++) {
         flop = (pmydata->commsize - 1) * (double)imyproblemSize;
         dresults[i+1] = (dtime[i]!=INVALID_MEASUREMENT) ? flop
            / dtime[i] : INVALID_MEASUREMENT;
      }
      /* flops for user-defined */
      i = 4;
      flop = (pmydata->commsize - 1) * 3 * (double)imyproblemSize;
      dresults[i+1] = (dtime[i]!=INVALID_MEASUREMENT) ? flop
         / dtime[i] : INVALID_MEASUREMENT;
      i = 5;
      flop = (pmydata->commsize - 1) * 26 * (double)imyproblemSize;
      dresults[i+1] = (dtime[i]!=INVALID_MEASUREMENT) ? flop
         / dtime[i] : INVALID_MEASUREMENT;
   }

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv) {
   mydata_t * pmydata = (mydata_t*)mdpv;

   if (in)
      free(in);
   if (out)
      free(out);

   if (invec)
      free(invec);
   if (outvec)
      free(outvec);

   if (inmat)
      free(inmat);
   if (outmat)
      free(outmat);

   if (dtime)
      free(dtime);

   MPI_Type_free(&myctypes[4]);
   MPI_Type_free(&myctypes[5]);
   if (myctypes)
      free(myctypes);

   MPI_Op_free(&myops[4]);
   MPI_Op_free(&myops[5]);
   if (myops)
      free(myops);

   if (pmydata)
      free(pmydata);
   return;
}


