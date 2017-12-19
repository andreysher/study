/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: pingpong.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/bandwidth/C/MPI/0/half2halfpingpong/pingpong.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: pairwise Send/Recv between two MPI-Prozesses>
 *         this file holds all the functions needed by the 
 *         benchit-interface
 *******************************************************************/

#include <stdio.h>

#include "pingpong.h"

void pingpong(unsigned int *from, unsigned int *to, void *mdpv, unsigned long int * msgsize)
{
  /* cast void* pointer */
  mydata_t * pmydata = (mydata_t *) mdpv;
  MPI_Status status;
  unsigned int loop;
  
  IDL(3, printf("rank %d, inside pingpong, send %ldbyte from %d to %d\n",pmydata->commrank,*msgsize,*from,*to));
  
  if(*msgsize == 0) 
      return;
  
  for(loop = 0; loop < pmydata->repeat; loop++)
  {
      if(pmydata->commrank == *from)
      {
        MPI_Send(pmydata->msg_string, *msgsize, MPI_BYTE, *to, 1, MPI_COMM_WORLD);
        MPI_Recv(pmydata->msg_string  , *msgsize, MPI_BYTE, *to, 1, MPI_COMM_WORLD, &status);
      }
      else if(pmydata->commrank == *to)
      {
        MPI_Recv(pmydata->msg_string, *msgsize, MPI_BYTE, *from, 1, MPI_COMM_WORLD, &status);
        MPI_Send(pmydata->msg_string, *msgsize, MPI_BYTE, *from, 1, MPI_COMM_WORLD);
      }
  }
}


