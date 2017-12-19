/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: pingpong.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/latencies/C/MPI/0/pingpong/pingpong.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: pairwise Send/Recv between two MPI-Prozesses>
 *         this file holds all the functions needed by the 
 *         benchit-interface
 *******************************************************************/

#include <stdio.h>

#include "pingpong.h"

void pingpong(int *from, int *to, void *mdpv)
{
  /* cast void* pointer */
  mydata_t * pmydata = (mydata_t *) mdpv;
  MPI_Status status;
  myinttype loop;

  IDL(3, printf("rank %d, inside pingpong\n", pmydata->commrank));
  
  if(pmydata->msgsize==0)
      return;
  
  for(loop = 0; loop < pmydata->repeat; loop++)
  {
      if(pmydata->commrank == *from)
      {
        MPI_Send(pmydata->buffer, pmydata->msgsize, MPI_BYTE, *to, 1, MPI_COMM_WORLD);
        MPI_Recv(pmydata->buffer  , pmydata->msgsize, MPI_BYTE, *to, 1, MPI_COMM_WORLD, &status);
      }
      else if(pmydata->commrank == *to)
      {
        MPI_Recv(pmydata->buffer, pmydata->msgsize, MPI_BYTE, *from, 1, MPI_COMM_WORLD, &status);
        MPI_Send(pmydata->buffer, pmydata->msgsize, MPI_BYTE, *from, 1, MPI_COMM_WORLD);
      }
  }
}


