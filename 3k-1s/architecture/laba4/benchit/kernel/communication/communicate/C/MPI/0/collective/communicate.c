/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: communicate.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/communicate/C/MPI/0/collective/communicate.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare the needed time for different MPI communication methodes
 *******************************************************************/

#include "communicate.h"
#include "interface.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void communicate(int rank, int size, int numfct, float * sbuffer,
   float * rbuffer, int entries_in_buffer, int * sendcounts,
   int * displs, double * dtime) {
   double t0=0, t1=0;

   switch (numfct) {
      case 0:
         t0 = bi_gettime();
         MPI_Bcast(sbuffer, entries_in_buffer, MPI_FLOAT, 0,
            MPI_COMM_WORLD);
         t1 = bi_gettime();
         break;
      case 1:
         t0 = bi_gettime();
         MPI_Scatter(sbuffer, entries_in_buffer, MPI_FLOAT, rbuffer,
            entries_in_buffer, MPI_FLOAT, 0, MPI_COMM_WORLD);
         t1 = bi_gettime();
         break;
      case 2:
         t0 = bi_gettime();
         MPI_Scatterv(sbuffer, sendcounts, displs, MPI_FLOAT,
            rbuffer, entries_in_buffer, MPI_FLOAT, 0, MPI_COMM_WORLD);
         t1 = bi_gettime();
         break;
      case 3:
         t0 = bi_gettime();
         MPI_Gather(sbuffer, entries_in_buffer, MPI_FLOAT, rbuffer,
            entries_in_buffer, MPI_FLOAT, 0, MPI_COMM_WORLD);
         t1 = bi_gettime();
         break;
      case 4:
         t0 = bi_gettime();
         MPI_Gatherv(sbuffer, entries_in_buffer, MPI_FLOAT, rbuffer,
            sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
         t1 = bi_gettime();
         break;
      case 5:
         t0 = bi_gettime();
         MPI_Allgather(sbuffer, entries_in_buffer, MPI_FLOAT,
            rbuffer, entries_in_buffer, MPI_FLOAT, MPI_COMM_WORLD);
         t1 = bi_gettime();
         break;
      case 6:
         t0 = bi_gettime();
         MPI_Allgatherv(sbuffer, entries_in_buffer, MPI_FLOAT,
            rbuffer, sendcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
         t1 = bi_gettime();
         break;
      case 7:
         t0 = bi_gettime();
         MPI_Alltoall(sbuffer, entries_in_buffer, MPI_FLOAT, rbuffer,
            entries_in_buffer, MPI_FLOAT, MPI_COMM_WORLD);
         t1 = bi_gettime();
         break;
      case 8:
         t0 = bi_gettime();
         MPI_Alltoallv(sbuffer, sendcounts, displs, MPI_FLOAT,
            rbuffer, sendcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
         t1 = bi_gettime();
         break;
      default:
         IDL(0, printf("\nError: selectet case is not available!"));
   }

   if (rank==0) {
      dtime[0] = t1 - t0 - dTimerOverhead;
   }
}


