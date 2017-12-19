/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: roundtrip.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/bandwidth/C/MPI/0/roundtrip/roundtrip.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Measure MPI bandwith for a round trip send algorithm
 *******************************************************************/


#include "roundtrip.h"

void entry_(void *mdpv, uint64_t * size) {
   mydata_t *pmydata = (mydata_t *) mdpv;

   if ((*size) == 0)
      return;
   else
      mpiroundtrip_(pmydata->sendbuf, pmydata->recvbuf, size,
                    &(pmydata->commrank), &(pmydata->commsize));
}

void mpiroundtrip_(void *send, void *recv, uint64_t * size, myinttype * rank,
                   myinttype * commsize) {
   MPI_Request receiverstates[2];
   MPI_Status requeststates[2];
   myinttype next = ((*rank) + 1) % (*commsize);
   myinttype prev = ((*rank) - 1 + (*commsize)) % (*commsize);
   myinttype nexttag = (myinttype)(*size + next);
   myinttype prevtag = (myinttype)(*size + *rank);

   IDL(2, printf("Rank: %d; Next: %d; Prev: %d;\n", *rank, next, prev));

   IDL(2, printf("Initiate send to %d (%d)\n", next, *rank));
   MPI_Isend(send, *size, MPI_CHAR, next, nexttag, MPI_COMM_WORLD,
             &receiverstates[0]);
   IDL(2, printf("Initiate receive from %d (%d)\n", prev, *rank));
   MPI_Irecv(recv, *size, MPI_CHAR, prev, prevtag, MPI_COMM_WORLD,
             &receiverstates[1]);
   IDL(2, printf("Wait for completion (%d)\n", *rank));
   MPI_Waitall(2, receiverstates, requeststates);
   IDL(2, printf("Complete (%d)\n", *rank));
}
