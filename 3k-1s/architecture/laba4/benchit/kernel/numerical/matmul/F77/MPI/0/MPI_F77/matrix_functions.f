CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C BenchIT - Performance Measurement for Scientific Applications
C Contact: developer@benchit.org
C
C $Id: matrix_functions.f 1 2009-09-11 12:26:19Z william $
C $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/F77/MPI/0/MPI_F77/matrix_functions.f $
C For license details see COPYING in the package base directory
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Kernel: Matrix multiply (Fortran 77) with MPI
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

        SUBROUTINE matrixinit( a, b, sizeall, sizeone)
        INTEGER sizeall, sizeone, i, j
        double precision a(sizeall,sizeall), b(sizeall,sizeone)
        double precision fill
        fill=0.0
        do 10 j=1,sizeall
           do 10 i=1,sizeall
              a(i,j)=1.0
10      continue
        call matrixfill( b, sizeall, sizeone, fill)
        RETURN
        END
C
        SUBROUTINE matrixfill( a, rows, cols, value)
        INTEGER rows, cols
        double precision a(rows,cols), value
        do 20 j=1,cols
           do 20 i=1,rows
              a(i,j)=value
20      continue
        RETURN
        END
C
        SUBROUTINE matrixprint( a, rows, cols)
        INTEGER rows, cols, i, j
        double precision a(rows,cols)
        do 30 j=1,rows
           WRITE(*,*) (a(j,i), i=1,cols)
30      continue  
        RETURN
        END

