/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: reduce.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/communication/reduction/C/MPI/0/reduce/reduce.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare the flops for different MPI reduction methodes
 *******************************************************************/

#include "reduce.h"
#include "interface.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

//void vecadd(Vector *invec, Vector *outvec, int *len, MPI_Datatype *dptr){
void vecadd(void *invec, void *outvec, int *len, MPI_Datatype *dptr) {
   int i;
   Vector *in, *out;

   in = (Vector*)invec;
   out = (Vector*)outvec;

   for (i=0; i<*len; i++) {
      out[i].x1 += in[i].x1;
      out[i].x2 += in[i].x2;
      out[i].x3 += in[i].x3;
   }
}

void matmul(void *inmat, void *outmat, int *len, MPI_Datatype *dptr) {
   float s1, s2, s3, s4, s5, s6, s7, s8;
   float m1, m2, m3, m4, m5, m6, m7;
   float t1, t2;
   int i;
   Matrix *in, *out;

   in = (Matrix*)inmat;
   out = (Matrix*)outmat;

   for (i=0; i<*len; i++) {
      /* Winograd algorithm for 2x2-matricies (http://www.f.kth.se/~f95-eeh/exjobb/background.html)*/
      s1 = in[i].a21 + in[i].a22;
      s2 = s1 - in[i].a11;
      s3 = in[i].a11 - in[i].a21;
      s4 = in[i].a12 - s2;
      s5 = out[i].a12 - out[i].a11;
      s6 = out[i].a22 - s5;
      s7 = out[i].a22 - out[i].a12;
      s8 = s6 - out[i].a21;

      m1 = s2 * s6;
      m2 = in[i].a11 * out[i].a11;
      m3 = in[i].a12 * out[i].a21;
      m4 = s3 * s7;
      m5 = s1 * s5;
      m6 = s4 * out[i].a22;
      m7 = in[i].a22 * s8;

      t1 = m1 + m2;
      t2 = t1 + m4;

      out[i].a11 = m2 + m3;
      out[i].a12 = t1 + m5 + m6;
      out[i].a21 = t2 - m7;
      out[i].a22 = t2 + m5;
   }
}

void reduce(int rank, int size, void *sendbuff, void *recvbuff,
   int sendbuffsize, MPI_Datatype *dptr, MPI_Op *op, double *dtime) {
   double t0=0, t1=0;

   t0 = bi_gettime();
   MPI_Reduce(sendbuff, recvbuff, sendbuffsize, dptr[0], op[0], 0,
      MPI_COMM_WORLD);
   t1 = bi_gettime();

   if (rank==0) {
      dtime[0] = t1 - t0 - dTimerOverhead;
   }
}

