CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C BenchIT - Performance Measurement for Scientific Applications
C Contact: developer@benchit.org
C
C $Id: mpi_functions.f 1 2009-09-11 12:26:19Z william $
C $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/F77/MPI/0/MPI_F77/mpi_functions.f $
C For license details see COPYING in the package base directory
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Kernel: Matrix multiply (Fortran 77) with MPI
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C       *************************************************************
C       fills some COMMON variables concerning mpi and the binary
C       tree; childX are the id's of the childs in the tree;
C       childcountX is the #childs for childX
C       *************************************************************
	SUBROUTINE mpivariables
	INCLUDE 'mpif.h'
	INTEGER childs
	EXTERNAL childs
	INTEGER size, rank, child0, child1
	INTEGER childcount0, childcount1, daddy
	INTEGER ierror
	COMMON size, rank, child0, child1, daddy
	COMMON childcount0, childcount1
	call MPI_COMM_SIZE( MPI_COMM_WORLD, size, ierror)
	call MPI_COMM_RANK( MPI_COMM_WORLD, rank, ierror)
	child0=rank*2+1
	child1=rank*2+2
	daddy=(rank-1)/2
	if (child0.lt.size) then
	  childcount0=childs( child0, size)
	  WRITE(*,*) 'left call childs_',rank,child0,childcount0
	else
	  childcount0=0
	end if
	if (child1.lt.size) then
	  childcount1=childs( child1, size)
	  WRITE(*,*) 'right call childs_',rank,child0,childcount1
	else
	  childcount1=0
	end if
	RETURN
	END
C
C       *********************************************************
C       Broadcast of 'bytes' bytes over binary tree; function
C       should only be used after a call to mpi_variables !!
C       *********************************************************
	SUBROUTINE mpibinarybroadcast ( ptr, bytes)
	INCLUDE 'mpif.h'
	INTEGER size,child0,child1,rank,bytes
	COMMON size, rank, child0, child1, daddy
	CHARACTER ptr(size)
	INTEGER status(MPI_STATUS_SIZE), ierror
C       now we can start
	if (rank.eq.0) then
	   if (child0.lt.size) then
	      call MPI_Send ( ptr, bytes, MPI_CHAR, child0, 1,
     &		              MPI_COMM_WORLD, ierror)
	   end if
	   if (child1.lt.size) then
	      call MPI_Send ( ptr, bytes, MPI_CHAR, child1, 1,
     &		              MPI_COMM_WORLD, ierror)
	   end if
	else
	   call MPI_Recv ( ptr, bytes, MPI_CHAR, daddy, 1,
     &			   MPI_COMM_WORLD, status, ierror)

	   if (child0.lt.size) then
	      call MPI_Send ( ptr, bytes, MPI_CHAR, child0, 1,
     &		              MPI_COMM_WORLD, ierror)
	   end if
	   if (child1.lt.size) then
	      call MPI_Send ( ptr, bytes, MPI_CHAR, child1, 1,
     &		              MPI_COMM_WORLD, ierror)
	   end if
	end if
	RETURN
	END
