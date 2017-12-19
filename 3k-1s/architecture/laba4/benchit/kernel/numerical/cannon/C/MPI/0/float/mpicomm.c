/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: mpicomm.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/cannon/C/MPI/0/float/mpicomm.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: a MPI version of matrix-matrix multiplication
 *         (cannon algotithm)
 *******************************************************************/

#include "mpicomm.h"

/*
* this function creates a new communicator for the program, thru that the program can handle processors with a empty matrix
* and the workload for the processors is better shared 
*/
void createMyMPIComm(int globalsize, int m, int n) {
   int num_ranks = 0, i = 0, save = 0, test;
   int *ranks = NULL;

   IDL(INFO,
       printf("\n---->Entered createMyMPIComm() for globalrank=%i\n",
              globalrank));

   size = globalsize;
   ranks = (int *)calloc(globalsize, sizeof(int));
   for (i = 0; i < globalsize; i++) {
      if (m * n < size) {
         size--;
         ranks[i] = size;
         num_ranks++;
      } else {
         save = i;
         break;
      }
   }

   for (i = save + 1; i < globalsize; i++) {
      test = ceil(sqrt(size));
      test *= test;
      if (test != size) {
         size--;
         ranks[i] = size;
         num_ranks++;
      } else
         break;
   }

   MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);
   MPI_Group_excl(MPI_GROUP_WORLD, num_ranks, ranks, &mygroup);
   MPI_Comm_create(MPI_COMM_WORLD, mygroup, &mycomm);

   MPI_Group_rank(mygroup, &rank);

   IDL(INFO,
       printf("\n<----Exit createMyMPIComm() for globalrank=%i\n", globalrank));
}

/*
* clean up the created communicator
*/
void freeMyMPIComm() {
   IDL(INFO,
       printf("\n---->Entered freeMyMPIComm() for globalrank=%i\n",
              globalrank));

   if (rank != MPI_UNDEFINED)
      MPI_Comm_free(&mycomm);
   MPI_Group_free(&MPI_GROUP_WORLD);
   MPI_Group_free(&mygroup);

   IDL(INFO,
       printf("\n<----Exit freeMyMPIComm() for globalrank=%i\n", globalrank));
}

DT **initialShift(DT * matrixA, int *mA, int *nA, DT * matrixB, int *mB,
                  int *nB, GridPosition gridpos, int cpu_x, int cpu_y) {
   int send_mn[2] = {0, 0}, recv_mn[2] = {0, 0}, dest = 0, source = 0;
   DT *tempMatrixA = NULL, *tempMatrixB = NULL;
   DT **pointer = NULL;
   MPI_Status status;

   IDL(INFO,
       printf("\n---->Entered initialShift() for globalrank=%i\n", globalrank));

   pointer = (DT **) malloc(2 * sizeof(DT *));
   pointer[0] = NULL;
   pointer[1] = NULL;

   /* makes the left shift, thru gridpos.row times, in matrix A for all rows >
    * 0 */
   if (gridpos.row > 0) {
      dest = (rank - gridpos.row * cpu_y + size) % size;
      source = (rank + gridpos.row * cpu_y) % size;

      if (rank < source) {
         send_mn[0] = *mA;
         send_mn[1] = *nA;
         IDL(INFO, printf("\nglobalrank=%i  dest_a_1=%i\n", globalrank, dest));
         MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
         MPI_Send(matrixA, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);

         IDL(INFO,
             printf("\nglobalrank=%i  source_a_1=%i\n", globalrank, source));
         MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
         tempMatrixA = createMatrix(recv_mn[0], recv_mn[1]);
         MPI_Recv(tempMatrixA, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
                  mycomm, &status);
      } else {
         IDL(INFO,
             printf("\nglobalrank=%i  source_a_2=%i\n", globalrank, source));
         MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
         tempMatrixA = createMatrix(recv_mn[0], recv_mn[1]);
         MPI_Recv(tempMatrixA, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
                  mycomm, &status);

         send_mn[0] = *mA;
         send_mn[1] = *nA;
         IDL(INFO, printf("\nglobalrank=%i  dest_a_2=%i\n", globalrank, dest));
         MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
         MPI_Send(matrixA, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);
      }

      *mA = recv_mn[0];
      *nA = recv_mn[1];
      clearMatrix(matrixA);
   }

   /* makes the upward shift, thru gridpos.column times, in matrix B for all
    * columns > 0 */
   if (gridpos.column > 0) {
      dest = gridpos.column * cpu_y + (rank - gridpos.column) % cpu_y;
      source = gridpos.column * cpu_y + (rank + gridpos.column) % cpu_y;

      if (rank < source) {
         send_mn[0] = *mB;
         send_mn[1] = *nB;
         dest = gridpos.column * cpu_y + (rank - gridpos.column) % cpu_y;
         IDL(INFO, printf("\nglobalrank=%i  dest_b_1=%i\n", globalrank, dest));
         MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
         MPI_Send(matrixB, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);

         IDL(INFO,
             printf("\nglobalrank=%i  source_b_1=%i\n", globalrank, source));
         MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
         tempMatrixB = createMatrix(recv_mn[0], recv_mn[1]);
         MPI_Recv(tempMatrixB, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
                  mycomm, &status);
      } else {
         IDL(INFO,
             printf("\nglobalrank=%i  source_b_2=%i\n", globalrank, source));
         MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
         tempMatrixB = createMatrix(recv_mn[0], recv_mn[1]);
         MPI_Recv(tempMatrixB, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
                  mycomm, &status);

         send_mn[0] = *mB;
         send_mn[1] = *nB;
         dest = gridpos.column * cpu_y + (rank - gridpos.column) % cpu_y;
         IDL(INFO, printf("\nglobalrank=%i  dest_b_2=%i\n", globalrank, dest));
         MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
         MPI_Send(matrixB, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);
      }

      *mB = recv_mn[0];
      *nB = recv_mn[1];
      clearMatrix(matrixB);
   }

   pointer[0] = tempMatrixA;
   pointer[1] = tempMatrixB;

   IDL(INFO,
       printf("\n<----Exit initialShift() for globalrank=%i\n", globalrank));

   return pointer;
}

DT **nextShift(DT * matrixA, int *mA, int *nA, DT * matrixB, int *mB, int *nB,
               GridPosition gridpos, int cpu_x, int cpu_y) {
   int send_mn[2] = {0, 0}, recv_mn[2] = {0, 0}, dest = 0, source = 0;
   DT *tempMatrixA = NULL, *tempMatrixB = NULL;
   DT **pointer = NULL;
   MPI_Status status;

   IDL(INFO,
       printf("\n---->Entered nextShift() for globalrank=%i\n", globalrank));

   pointer = (DT **) malloc(2 * sizeof(DT *));
   pointer[0] = NULL;
   pointer[1] = NULL;

   /* makes the 1-left shift in matrix A */
   if (gridpos.column == 0) {
      send_mn[0] = *mA;
      send_mn[1] = *nA;
      dest = (rank - cpu_y + size) % size;
      IDL(INFO, printf("\nglobalrank=%i  dest_a_1=%i\n", globalrank, dest));
      MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
      MPI_Send(matrixA, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);
   } else {
      source = (rank + cpu_y) % size;
      IDL(INFO, printf("\nglobalrank=%i  source_a_2=%i\n", globalrank, source));
      MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
      tempMatrixA = createMatrix(recv_mn[0], recv_mn[1]);
      MPI_Recv(tempMatrixA, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
               mycomm, &status);

      send_mn[0] = *mA;
      send_mn[1] = *nA;
      dest = (rank - cpu_y + size) % size;
      IDL(INFO, printf("\nglobalrank=%i  dest_a_2=%i\n", globalrank, dest));
      MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
      MPI_Send(matrixA, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);
   }
   if (gridpos.column == 0) {
      source = (rank + cpu_y) % size;
      IDL(INFO, printf("\nglobalrank=%i  source_a_1=%i\n", globalrank, source));
      MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
      tempMatrixA = createMatrix(recv_mn[0], recv_mn[1]);
      MPI_Recv(tempMatrixA, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
               mycomm, &status);
   }
   *mA = recv_mn[0];
   *nA = recv_mn[1];
   clearMatrix(matrixA);
   pointer[0] = tempMatrixA;

   /* makes the 1-upward shift in matrix B */
   if (gridpos.row == 0) {
      send_mn[0] = *mB;
      send_mn[1] = *nB;
      dest = gridpos.column * cpu_y + (rank - 1 + cpu_y) % cpu_y;
      IDL(INFO, printf("\nglobalrank=%i  dest_b_1=%i\n", globalrank, dest));
      MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
      MPI_Send(matrixB, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);
   } else {
      source = gridpos.column * cpu_y + (rank + 1) % cpu_y;
      IDL(INFO, printf("\nglobalrank=%i  source_b_2=%i\n", globalrank, source));
      MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
      tempMatrixB = createMatrix(recv_mn[0], recv_mn[1]);
      MPI_Recv(tempMatrixB, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
               mycomm, &status);

      send_mn[0] = *mB;
      send_mn[1] = *nB;
      dest = gridpos.column * cpu_y + (rank - 1 + cpu_y) % cpu_y;
      IDL(INFO, printf("\nglobalrank=%i  dest_b_2=%i\n", globalrank, dest));
      MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
      MPI_Send(matrixB, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);
   }
   if (gridpos.row == 0) {
      source = gridpos.column * cpu_y + (rank + 1) % cpu_y;
      IDL(INFO, printf("\nglobalrank=%i  source_b_1=%i\n", globalrank, source));
      MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
      tempMatrixB = createMatrix(recv_mn[0], recv_mn[1]);
      MPI_Recv(tempMatrixB, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
               mycomm, &status);
   }
   *mB = recv_mn[0];
   *nB = recv_mn[1];
   clearMatrix(matrixB);
   pointer[1] = tempMatrixB;

   IDL(INFO, printf("\n<----Exit nextShift() for globalrank=%i\n", globalrank));

   return pointer;
}

DT **undoShift(DT * matrixA, int *mA, int *nA, DT * matrixB, int *mB, int *nB,
               GridPosition gridpos, int cpu_x, int cpu_y) {
   int send_mn[2] = {0, 0}, recv_mn[2] = {0, 0}, dest = 0, source = 0;
   DT *tempMatrixA = NULL, *tempMatrixB = NULL;
   DT **pointer = NULL;
   MPI_Status status;

   IDL(INFO,
       printf("\n---->Entered undoShift() for globalrank=%i\n", globalrank));

   pointer = (DT **) malloc(2 * sizeof(DT *));
   pointer[0] = NULL;
   pointer[1] = NULL;

   /* makes the right shift in matrix A, to get the initial matrix */
   if (gridpos.row > 0) {
      dest = (rank + gridpos.row * cpu_y) % size;
      source = (rank - gridpos.row * cpu_y + size) % size;

      if (rank < source) {
         send_mn[0] = *mA;
         send_mn[1] = *nA;
         IDL(INFO, printf("\nglobalrank=%i  dest_a_1=%i\n", globalrank, dest));
         MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
         MPI_Send(matrixA, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);

         IDL(INFO,
             printf("\nglobalrank=%i  source_a_1=%i\n", globalrank, source));
         MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
         tempMatrixA = createMatrix(recv_mn[0], recv_mn[1]);
         MPI_Recv(tempMatrixA, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
                  mycomm, &status);
      } else {
         IDL(INFO,
             printf("\nglobalrank=%i  source_a_2=%i\n", globalrank, source));
         MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
         tempMatrixA = createMatrix(recv_mn[0], recv_mn[1]);
         MPI_Recv(tempMatrixA, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
                  mycomm, &status);

         send_mn[0] = *mA;
         send_mn[1] = *nA;
         IDL(INFO, printf("\nglobalrank=%i  dest_a_2=%i\n", globalrank, dest));
         MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
         MPI_Send(matrixA, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);
      }

      *mA = recv_mn[0];
      *nA = recv_mn[1];
      clearMatrix(matrixA);
   }

   /* makes the downward shift in matrix B, to get the initial matrix */
   if (gridpos.column > 0) {
      dest = gridpos.column * cpu_y + (rank + gridpos.column) % cpu_y;
      source = gridpos.column * cpu_y + (rank - gridpos.column) % cpu_y;

      if (rank < source) {
         send_mn[0] = *mB;
         send_mn[1] = *nB;
         IDL(INFO, printf("\nglobalrank=%i  dest_b_1=%i\n", globalrank, dest));
         MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
         MPI_Send(matrixB, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);

         IDL(INFO,
             printf("\nglobalrank=%i  source_b_1=%i\n", globalrank, source));
         MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
         tempMatrixB = createMatrix(recv_mn[0], recv_mn[1]);
         MPI_Recv(tempMatrixB, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
                  mycomm, &status);
      } else {
         IDL(INFO,
             printf("\nglobalrank=%i  source_b_2=%i\n", globalrank, source));
         MPI_Recv(recv_mn, 2, MPI_iDT, source, tag, mycomm, &status);
         tempMatrixB = createMatrix(recv_mn[0], recv_mn[1]);
         MPI_Recv(tempMatrixB, recv_mn[0] * recv_mn[1], MPI_DT, source, tag,
                  mycomm, &status);

         send_mn[0] = *mB;
         send_mn[1] = *nB;
         IDL(INFO, printf("\nglobalrank=%i  dest_b_2=%i\n", globalrank, dest));
         MPI_Send(send_mn, 2, MPI_iDT, dest, tag, mycomm);
         MPI_Send(matrixB, send_mn[0] * send_mn[1], MPI_DT, dest, tag, mycomm);
      }

      *mB = recv_mn[0];
      *nB = recv_mn[1];
      clearMatrix(matrixB);
   }

   pointer[0] = tempMatrixA;
   pointer[1] = tempMatrixB;

   IDL(INFO, printf("\n<----Exit undoShift() for globalrank=%i\n", globalrank));

   return pointer;
}

