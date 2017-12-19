/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: communicate.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/communicate/C/MPI/0/nonblocking/communicate.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare the needed time for different MPI communication methodes
 *******************************************************************/

#include "communicate.h"
#include "interface.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

void communicate(int rank, int size, int numfct, float * buffer,
   int entries_in_buffer, double * dtime) {
   int i=0;
   double t0=0, t1=0, t2=0, tmin=DBL_MAX, tmax=0, tmean=0;
   MPI_Status status;
   MPI_Request request;

   switch (numfct) {
      case 0:
         if (rank==0) {
            t0 = bi_gettime();
            for (i=1; i<size; i++) {
               t1 = bi_gettime();
               MPI_Isend(buffer, entries_in_buffer, MPI_FLOAT, i, i
                  *10+1, MPI_COMM_WORLD, &request);
               MPI_Wait(&request, &status);
               t2 = bi_gettime();
               tmin = (t2-t1 < tmin) ? t2-t1 : tmin;
               tmax = (t2-t1 > tmax) ? t2-t1 : tmax;
            }
            tmean = bi_gettime() - t0;
         } else {
            MPI_Irecv(buffer, entries_in_buffer, MPI_FLOAT, 0, rank
               *10+1, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
         }
         break;
      case 1:
         if (rank==0) {
            t0 = bi_gettime();
            for (i=1; i<size; i++) {
               t1 = bi_gettime();
               MPI_Issend(buffer, entries_in_buffer, MPI_FLOAT, i, i
                  *10+2, MPI_COMM_WORLD, &request);
               MPI_Wait(&request, &status);
               t2 = bi_gettime();
               tmin = (t2-t1 < tmin) ? t2-t1 : tmin;
               tmax = (t2-t1 > tmax) ? t2-t1 : tmax;
            }
            tmean = bi_gettime() - t0;
         } else {
            MPI_Irecv(buffer, entries_in_buffer, MPI_FLOAT, 0, rank
               *10+2, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
         }
         break;
      case 2:
         if (rank==0) {
            t0 = bi_gettime();
            for (i=1; i<size; i++) {
               t1 = bi_gettime();
               MPI_Irsend(buffer, entries_in_buffer, MPI_FLOAT, i, i
                  *10+3, MPI_COMM_WORLD, &request);
               MPI_Wait(&request, &status);
               t2 = bi_gettime();
               tmin = (t2-t1 < tmin) ? t2-t1 : tmin;
               tmax = (t2-t1 > tmax) ? t2-t1 : tmax;
            }
            tmean = bi_gettime() - t0;
         } else {
            MPI_Irecv(buffer, entries_in_buffer, MPI_FLOAT, 0, rank
               *10+3, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
         }
         break;
      case 3:
         /* attention: in principle the measurement of Ibsend should only show the memory bandwidth on rank 0
          *            and NOT the communication bandwidth (it's is measured here only because of completeness) */
         if (rank==0) {
            t0 = bi_gettime();
            for (i=1; i<size; i++) {
               t1 = bi_gettime();
               MPI_Ibsend(buffer, entries_in_buffer, MPI_FLOAT, i, i
                  *10+4, MPI_COMM_WORLD, &request);
               MPI_Wait(&request, &status);
               t2 = bi_gettime();
               tmin = (t2-t1 < tmin) ? t2-t1 : tmin;
               tmax = (t2-t1 > tmax) ? t2-t1 : tmax;
            }
            tmean = bi_gettime() - t0;
         } else {
            MPI_Irecv(buffer, entries_in_buffer, MPI_FLOAT, 0, rank
               *10+4, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
         }
         break;
      default:
         IDL(0, printf("\nError: selectet case is not available!"));
   }

   if (rank==0) {
      dtime[0] = tmin - dTimerOverhead;
      dtime[1] = tmax - dTimerOverhead;
      dtime[2] = tmean - dTimerOverhead - (2 * (size-1)
         * dTimerOverhead);
   }
}


