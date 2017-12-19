/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: binarybroadcast.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/bandwidth/C/MPI/0/binarybroadcast/binarybroadcast.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: bandwidth for a mpi broadcast implemented with a binary tree and send
 *******************************************************************/

#include "binarybroadcast.h"
#include "interface.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

void mpibinarybroadcast_(void *ptr, int *size, int *rank,
   int *commsize) {
   MPI_Status status;
   MPI_Request receiverstates[2];
   MPI_Status requeststates[2];
   int daddy = (*rank - 1) >> 1;
   int child1 = (*rank) << 1, child2;
   int count = 0;

   child1++;
   child2 = child1 + 1;

   IDL(2, printf("Rank: %d; Child1: %d; Child2: %d; Daddy: %d\n",
      *rank, child1, child2, daddy));

   if (*rank == 0) {
      /* nur senden und selbes Paket wieder empfangen */
      if (child1 < *commsize) {
         IDL(2, printf("Sending to %d (%d)\n", child1, *rank));
         MPI_Send(ptr, *size, MPI_CHAR, child1, 1, MPI_COMM_WORLD);
         IDL(2,
            printf("Sending to %d complete (%d)\n", child1, *rank));

         IDL(2, printf("Init receive from first child %d (%d)\n",
            child1, *rank));
         MPI_Irecv(ptr, *size, MPI_CHAR, MPI_ANY_SOURCE, 1,
            MPI_COMM_WORLD, &receiverstates[count++]);
      }
      if (child2 < *commsize) {
         IDL(2, printf("Sending to %d (%d)\n", child1, *rank));
         MPI_Send(ptr, *size, MPI_CHAR, child2, 1, MPI_COMM_WORLD);
         IDL(2,
            printf("Sending to %d complete (%d)\n", child1, *rank));

         IDL(2, printf("Init receive from second child %d (%d)\n",
            child2, *rank));
         MPI_Irecv(ptr, *size, MPI_CHAR, MPI_ANY_SOURCE, 1,
            MPI_COMM_WORLD, &receiverstates[count++]);
      }
      /* wait for response */
      IDL(2, printf("Wait for response, %d answers (%d)\n", count,
         *rank));
      MPI_Waitall(count, receiverstates, requeststates);
      IDL(2, printf("Got response (%d)\n", *rank));
   } else {
      /* erst empfangen und dann weitersenden */
      IDL(2, printf("Receiving from %d (%d)\n", daddy, *rank));
      MPI_Recv(ptr, *size, MPI_CHAR, MPI_ANY_SOURCE, 1,
         MPI_COMM_WORLD, &status);
      IDL(2, printf("Receive from %d complete (%d)\n", daddy, *rank));
      if (child1 < *commsize) {
         IDL(2, printf("Sending to %d (%d)\n", child1, *rank));
         MPI_Send(ptr, *size, MPI_CHAR, child1, 1, MPI_COMM_WORLD);
         IDL(2,
            printf("Sending to %d complete (%d)\n", child1, *rank));
      }
      if (child2 < *commsize) {
         IDL(2, printf("Sending to %d (%d)\n", child2, *rank));
         MPI_Send(ptr, *size, MPI_CHAR, child2, 1, MPI_COMM_WORLD);
         IDL(2,
            printf("Sending to %d complete (%d)\n", child2, *rank));
      }
      /* und den zweiten Teil auch noch -
       * alle Blaetter des binaeren Baumes senden wieder nach oben, bis
       * alles wieder bei root ankommt, Blatt ist jeder ohne Kinder */
      if (child1 >= *commsize) {
         IDL(2, printf("Sending to %d (%d)\n", daddy, *rank));
         MPI_Send(ptr, *size, MPI_CHAR, daddy, 1, MPI_COMM_WORLD);
         IDL(2, printf("Sending to %d complete (%d)\n", daddy, *rank));
      } else {
         /* wenn nicht Blatt, kriegen wir soviele Nachrichten, wie wir Kinder haben
          * und muessen selbst eine nach oben schicken */
         count = 0;
         if (child1 < *commsize) {
            IDL(2, printf("Init receive from %d (%d)\n", child1,
               *rank));
            MPI_Irecv(ptr, *size, MPI_CHAR, MPI_ANY_SOURCE, 1,
               MPI_COMM_WORLD, &receiverstates[count++]);
         }
         if (child2 < *commsize) {
            IDL(2, printf("Init receive from %d (%d)\n", child2,
               *rank));
            MPI_Irecv(ptr, *size, MPI_CHAR, MPI_ANY_SOURCE, 1,
               MPI_COMM_WORLD, &receiverstates[count++]);
         }
         IDL(2, printf("Wait for response, %d answers (%d)\n", count,
            *rank));
         MPI_Waitall(count, receiverstates, requeststates);
         IDL(2, printf("Got response for %d\n", *rank));

         IDL(2, printf("Sending to %d (%d)\n", daddy, *rank));
         MPI_Send(ptr, *size, MPI_CHAR, daddy, 1, MPI_COMM_WORLD);
         IDL(2, printf("Sending to %d complete (%d)\n", daddy, *rank));
      }
   }
}


