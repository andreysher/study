!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! BenchIT - Performance Measurement for Scientific Applications
! Contact: developer@benchit.org
!
! $Id: 2dPoisson.f90 1 2009-09-11 12:26:19Z william $
! $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/applications/MGV/F95/OpenMP/0/recursive/2dPoisson.f90 $
! For license details see COPYING in the package base directory
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Kernel: multigrid methode for 2D Poisson equation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


module debug
   implicit none

   LOGICAL, parameter :: INFO=.FALSE., INFOTIME=.FALSE.     ! is used for debug output
   INTEGER(kind=4) :: st, z1, z2

   interface
      subroutine f90fflush()
      end subroutine f90fflush
   end interface

end module debug

!************ modules ************


!************ main programm routines ************

module my2dPoisson
   !use memory
   use debug
   use omp_lib

   implicit none

   type matrixStruct
      REAL(kind=8), dimension(:, :), allocatable :: matrix
   end type matrixStruct

   contains

   subroutine my2dPoisson_without_sparse( level0, maxlevel0, outputform, v1, v2, &
                                          & w, L1, L2, time_for_MGV, omega, flop )
      ! in/out
      INTEGER(kind=4) :: level0, maxlevel0, outputform, v1, v2
      REAL(kind=8) :: w, L1, L2
      REAL(kind=8) :: time_for_MGV, omega, flop
      ! additional
      INTEGER(kind=4) :: iter, start, counter, i, M1M1, j, k, z3, z4
      REAL(kind=8) :: res, res2, old_res
      REAL(kind=8) :: t1, t2
      CHARACTER(len=30) :: str

      type(matrixStruct), dimension(:), allocatable :: struct
   
      REAL(kind=8), dimension(:), allocatable :: b, solution, r, x0
      
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "my2dPoisson_without_sparse( level0, maxlevel0, outputform, &
                    & v1, v2, w, L1, L2, time_for_MGV, omega, flop ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
   
      allocate( struct(1:level0), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the struct, status=", st, "!"
         call f90fflush()
         stop
      end if
      
      ! create the matrix for each level with a size constrained on the level
      do i=1, level0, 1
         M1M1 = (2**i + 1)**2
         write(str,fmt='(A,I0,A)') "struct(", i, ")%matrix"
         !CALL alloc( struct(i)%matrix, M1M1, M1M1, trim(str) )
         allocate( struct(i)%matrix(1:M1M1,1:M1M1), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,A,A,I0,A)') "Error: Allocation of matrix ", str,", status=", st, "!"
            call f90fflush()
            stop
         end if
         !struct(i)%matrix = 0
         if( INFO ) call system_clock( z1 )
         !$omp parallel do shared(M1M1, struct) firstprivate(i) private(j)
         do j=1, M1M1, 1
            struct(i)%matrix(:,j) = 0
         end do
         !$omp end parallel do
         if( INFO ) call system_clock( z2 )
         if( INFO ) write(*,*) "time1 in clockcycle: ", z2-z1
      end do

      CALL initializeMatrixStruct( struct, level0 )
   
      if( INFO ) CALL printMatrixStruct( struct )

      !CALL alloc( b, N1N1, "b" )
      allocate( b(1:(2**level0+1)**2), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the vector b, status=", st, "!"
         call f90fflush()
         stop
      end if
      b = 0

      CALL initializeVectorB( b, level0, L1, L2 )
      if( INFO ) write(*,*) b
      
      counter = 0
      res = 0.0
      res2 = 0.0
      old_res = 1.0
   
      !CALL alloc( solution, (2**level0+1)**2, "solution" )
      allocate( solution(1:(2**level0+1)**2), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the vector solution, status=", st, "!"
         call f90fflush()
         stop
      end if
      solution = 0

      !CALL alloc( x0, (2**level0+1)**2, "x0" )
      allocate( x0(1:(2**level0+1)**2), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the vector x0, status=", st, "!"
         call f90fflush()
         stop
      end if
      x0 = 0

      !CALL alloc( r, (2**level0+1)**2, "r" )
      allocate( r(1:(2**level0+1)**2), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the vector r, status=", st, "!"
         call f90fflush()
         stop
      end if
      r = 0
   
      CALL bi_gettime( t1 )
      do
         CALL MGV( solution, struct, x0, b, level0, v1, v2, w )
         !r = b - struct(iter)%matrix * solution
         CALL MatVecMult( r, struct(level0)%matrix, solution )
         r = b - r

         res = maxval(abs(r))
         counter = counter + 1
         res2 = res2 + res / old_res
         old_res = res
   
         if( res<0.05 ) then
            exit
         else
            x0 = solution
         end if
      end do
      CALL bi_gettime( t2 )
   
      !CALL dealloc( r, "r" )
      deallocate( r, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector r, status=", st, "!"
         call f90fflush()
      end if

      !CALL dealloc( x0, "x0" )
      deallocate( x0, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector x0, status=", st, "!"
         call f90fflush()
      end if

      !CALL dealloc( b, "b" )
      deallocate( b, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector b, status=", st, "!"
         call f90fflush()
      end if
   
      if( outputform/=0 .AND. level0==maxlevel0 ) then
         !write(*,*) level0, maxlevel0
         CALL writeOutput( outputform, level0, solution, L1, L2 )
      end if
   
      time_for_MGV = t2-t1
      omega = res2 / counter
      flop = calcFlop(level0, counter, v1, v2)
      
      !CALL dealloc( solution, "solution" )
      deallocate( solution, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector solution, status=", st, "!"
         call f90fflush()
      end if

      !CALL removeMatrixStruct( struct )
      do i=1, level0, 1
         deallocate(struct(i)%matrix, stat=st)
         if( st/=0 ) then
            write(*,fmt='(A,I0,A,I0,A)') "Error: Deallocation of the matrix for level=", i, &
                                         & " not possible, status=", st, "!"
            call f90fflush()
         end if
      end do
   
      deallocate(struct, stat=st)
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of the struct not possible, status=", st, "!"
         call f90fflush()
      end if
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "my2dPoisson_without_sparse( level0, maxlevel0, outputform, v1, v2, & 
                    & w, L1, L2, time_for_MGV, omega, flop ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine my2dPoisson_without_sparse
   
   
   ! initialize for each level a matrix and safe in the struct
   ! boundary condition are also included
   subroutine initializeMatrixStruct( struct, level0 )
      ! in/out
      type(matrixStruct), dimension(:) :: struct
      INTEGER(kind=4) :: level0
      ! additional
      INTEGER(kind=4) :: i, j, M1, M1M1, column, temp, z3, z4
      REAL(kind=8) :: factor

      if( INFO .OR. INFOTIME ) then
         write(*,*) "initializeMatrixStruct( struct ) in"
         call system_clock( z3 )
         call f90fflush()
      end if

      ! fill the matrix with the diagonals scaled -1, 4, -1
      do i=1, level0, 1
         M1M1 = (2**i + 1)**2
         factor = ((2**i)**2)
         struct(i)%matrix(1:2,1) = factor * (/ 4.0, -1.0 /)
         if( INFO ) call system_clock( z1 )
         !$omp parallel do shared(struct, M1M1, factor) firstprivate(i) private(column)
         do column=2, M1M1-1, 1
            struct(i)%matrix(column-1:column+1,column) = factor * (/ -1.0, 4.0, -1.0 /)
         end do
         !$omp end parallel do
         if( INFO ) call system_clock( z2 )
         if( INFO ) write(*,*) "time2 in clockcycle: ", z2-z1
         struct(i)%matrix(M1M1-1:M1M1,M1M1) = factor * (/ -1.0, 4.0 /)
      end do

      ! generate the additional information for a 5-point-star for all matries and add the boundary condition
      if( INFO ) call system_clock( z1 )
      !$omp parallel do shared(struct, level0) private(M1, M1M1, temp, j, i)
      do i=1, level0, 1
         ! 5-point-star
         !struct(i)%matrix = ((2**i)**2) * struct(i)%matrix
         
         M1 = 2**i+1
         M1M1 = M1**2
         ! temp = -1/h2
         temp = -((2**i)**2)
         do j=M1+1, M1M1-M1
            struct(i)%matrix(j,(/ j-M1, j+M1 /)) = (/ temp, temp /)
         end do

         ! boundary condition
         struct(i)%matrix(1,(/ 1, 2, 1+M1/)) = 0
         struct(i)%matrix(1,1) = 1
         do j=2, M1, 1
            struct(i)%matrix(j,(/ j-1, j, j+1, j+M1 /)) = 0
            struct(i)%matrix(j,j) = 1
         end do
         do j=M1M1-(M1-1), M1M1-1, 1
            struct(i)%matrix(j,(/ j-M1, j-1, j, j+1 /)) = 0
            struct(i)%matrix(j,j) = 1
         end do
         struct(i)%matrix(M1M1,(/ M1M1-M1, M1M1-1, M1M1 /)) = 0
         struct(i)%matrix(M1M1,M1M1) = 1
         do j=M1+1, M1M1-M1, M1
            struct(i)%matrix(j,(/ j-M1, j-1, j, j+1, j+M1 /)) = 0
            struct(i)%matrix(j,j) = 1
         end do
         do j=M1+M1, M1M1-M1, M1
            struct(i)%matrix(j,(/ j-M1, j-1, j, j+1, j+M1 /)) = 0
            struct(i)%matrix(j,j) = 1
         end do
      end do
      !$omp end parallel do
      if( INFO ) call system_clock( z2 )
      if( INFO ) write(*,*) "time4 in clockcycle: ", z2-z1
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "initializeMatrixStruct( struct ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine initializeMatrixStruct
   
   
   subroutine initializeVectorB( b, level, L1, L2 )
      ! in/out
      REAL(kind=8), dimension(:) :: b
      INTEGER(kind=4) :: level
      REAL(kind=8) :: L1, L2
      ! additional
      INTEGER(kind=4) :: i, j, N1, N1N1, z3, z4
      LOGICAL :: test=.FALSE.
      REAL(kind=8), dimension(:), allocatable :: x1, x2
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "initializeVectorB( b, level, L1, L2 ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
   
      !if( allocated(b) ) CALL dealloc( b, "b" )
   
      N1 = 2**level + 1
      N1N1 = N1**2
      !CALL alloc( b, N1N1, "b" )
   
      if( test ) then
         !b((j-1)*((N1-1)+1)+i) = 2
         b = 2
      else
         !CALL alloc( x1, N1, "x1 in initB" )
         allocate( x1(1:N1), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of the vector x1 in initializeVectorB, status=", st, "!"
            call f90fflush()
            stop
         end if
         x1 = 0

         !CALL alloc( x2, N1, "x2 in initB" )
         allocate( x2(1:N1), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of the vector x2 in initializeVectorB, status=", st, "!"
            call f90fflush()
            stop
         end if
         x2 = 0

         CALL get_x1( x1, N1, L1)
         CALL get_x2( x2, N1, L2)
   
         if( INFO ) call system_clock( z1 )
         !$omp parallel do shared(b, x1, x2, N1) private(i, j)
         do j=1, N1, 1
             do i=1, N1, 1
               !b((j-1)*((N1-1)+1)+i) = 2 * ( x1(i)*(1-(x1(i))) + x2(j)*(1-(x2(j))) )
               b((j-1)*N1+i) = 2 * ( x1(i)*(1-(x1(i))) + x2(j)*(1-(x2(j))) )
             end do
         end do
         !$omp end parallel do
         if( INFO ) call system_clock( z2 )
         if( INFO ) write(*,*) "time5 in clockcycle: ", z2-z1
   
         !CALL dealloc( x1, "x1 in initB" )
         deallocate( x1, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector x1 in initializeVectorB, status=", st, "!"
            call f90fflush()
         end if
   
         !CALL dealloc( x2, "x2 in initB" )
         deallocate( x2, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector x2 in initializeVectorB, status=", st, "!"
            call f90fflush()
         end if
      end if
   
      b(1:N1) = 0
      b(N1N1-(N1-1):N1N1) = 0
      b(N1+1:N1N1:N1) = 0
      b(N1:N1N1:N1) = 0
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "initializeVectorB( b, level, L1, L2 ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine initializeVectorB
   
   
   subroutine get_x1( x1, N1, L1 )
      ! in/out
      REAL(kind=8), dimension(:) :: x1
      INTEGER(kind=4) :: N1
      REAL(kind=8) :: L1
      ! additional
      INTEGER(kind=4) :: i, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "get_x1( x1, N1, L1 ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
   
      if( INFO ) call system_clock( z1 )
      !$omp parallel do shared(x1, N1, L1) private(i)
      do i=1, N1, 1
         x1(i) = ((i-1) * L1) / (N1-1)
      end do
      !$omp end parallel do
      if( INFO ) call system_clock( z2 )
      if( INFO ) write(*,*) "time6 in clockcycle: ", z2-z1
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "get_x1( x1, N1, L1 ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine get_x1
   
   
   subroutine get_x2( x2, N1, L2 )
      ! in/out
      REAL(kind=8), dimension(:) :: x2
      INTEGER(kind=4) :: N1
      REAL(kind=8) :: L2
      ! additional
      INTEGER(kind=4) :: i, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "get_x2( x2, N1, L2 ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
   
      if( INFO ) call system_clock( z1 )
      !$omp parallel do shared(x2, N1, L2) private(i)
      do i=1, N1, 1
         x2(i) = ((i-1) * L2) / (N1-1)
      end do
      !$omp end parallel do
      if( INFO ) call system_clock( z2 )
      if( INFO ) write(*,*) "time7 in clockcycle: ", z2-z1
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "get_x2( x2, N1, L2 ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine get_x2
   
   
   recursive subroutine MGV( x, struct, x0, b, level, v1, v2, w )
      ! in/out
      REAL(kind=8), dimension(:) :: x
      type(matrixStruct), dimension(:) :: struct
      REAL(kind=8), dimension(:) :: x0, b
      INTEGER(kind=4) :: level, v1, v2
      REAL(kind=8) :: w
      ! additional
      INTEGER(kind=4) :: row, n, n1, n1n1, tsize, z3, z4
      REAL(kind=8), dimension(:), allocatable :: null_vector, r, temp_vector, dx
      REAL(kind=8), dimension(:,:), allocatable :: temp_matrix1, temp_matrix2, inverse

      if( INFO .OR. INFOTIME ) then
         write(*,*) "MGV( solution, struct, x0, b, iter ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
   
      if( level==1 ) then
         n = (2**level+1)**2
         !CALL alloc( inverse, n, n, "inverse" )
         allocate( inverse(1:n,1:n), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of matrix inverse, status=", st, "!"
            call f90fflush()
            stop
         end if
         inverse = 0

         inverse = RESHAPE( (/ 1, 0, 0, 0, 0, 0, 0, 0, 0, &
                             & 0, 1, 0, 0, 1, 0, 0, 0, 0, &
                             & 0, 0, 1, 0, 0, 0, 0, 0, 0, &
                             & 0, 0, 0, 1, 1, 0, 0, 0, 0, &
                             & 0, 0, 0, 0, 1, 0, 0, 0, 0, &
                             & 0, 0, 0, 0, 1, 1, 0, 0, 0, &
                             & 0, 0, 0, 0, 0, 0, 1, 0, 0, &
                             & 0, 0, 0, 0, 1, 0, 0, 1, 0, &
                             & 0, 0, 0, 0, 0, 0, 0, 0, 1 /), (/9,9/) )
          inverse(5,2) = inverse(5,2)/4
          inverse(5,4) = inverse(5,4)/4
          inverse(5,5) = inverse(5,5)/16
          inverse(5,6) = inverse(5,6)/4
          inverse(5,8) = inverse(5,8)/4

         ! x = inverse * b
         CALL MatVecMult( x, inverse, b )
   
         !CALL dealloc( inverse, "inverse" )
         deallocate( inverse, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of matrix inverse, status=", st, "!"
            call f90fflush()
         end if
      else
         CALL JacobiRelax( x, struct(level)%matrix, x0, b, v1, w )
         
         !r = b - struct(level)%matrix * x
         !CALL alloc( r, size(x), "r in MGV" )
         allocate( r(1:size(x)), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of the vector r in MGV, status=", st, "!"
            call f90fflush()
            stop
         end if
         r = 0

         CALL MatVecMult( r, struct(level)%matrix, x )
         r = b - r

         tsize = int(sqrt(real(size(r))))         
         allocate( temp_matrix1(1:tsize,1:tsize), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of matrix temp_matrix1 for transform_vector_to_matrix, &
                                    & status=", st, "!"
            call f90fflush()
            stop
         end if
         temp_matrix1 = 0
         CALL transform_vector_to_matrix( temp_matrix1, r )
         deallocate( r, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector r after transform_vector_to_matrix, &
                                    & status=", st, "!"
            call f90fflush()
         end if
   
         tsize = 2**(level-1) + 1
         allocate( temp_matrix2(1:tsize,1:tsize), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of matrix temp_matrix2 for restriction, status=", st, "!"
            call f90fflush()
            stop
         end if
         temp_matrix2 = 0
         CALL restriction( temp_matrix2, temp_matrix1, level )
         deallocate( temp_matrix1, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of matrix temp_matrix1 after restriction, status=", st, "!"
            call f90fflush()
         end if
   
         tsize = size(temp_matrix2)
         allocate( temp_vector(1:tsize), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of the vector temp_vector for transform_matrix_to_vector, &
                                    & status=", st, "!"
            call f90fflush()
            stop
         end if
         temp_vector = 0
         CALL transform_matrix_to_vector( temp_vector, temp_matrix2 )
         deallocate( temp_matrix2, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of matrix temp_matrix2 after transform_matrix_to_vector, &
                                    & status=", st, "!"
            call f90fflush()
         end if
   
         n1 = 2**(level-1) + 1
         n1n1 = n1 * n1
         temp_vector(1:n1) = 0
         temp_vector(n1n1-(n1-1):n1n1) = 0
         temp_vector(n1+1:n1n1:n1) = 0
         temp_vector(n1:n1n1:n1) = 0
   
         !CALL alloc( dx, n1n1, "dx in MGV" )
         allocate( dx(1:n1n1), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of the vector dx in MGV, status=", st, "!"
            call f90fflush()
            stop
         end if
         dx = 0

         !CALL alloc( null_vector, n1n1, "null_vector in MGV" )
         allocate( null_vector(1:n1n1), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of the vector null_vector in MGV, status=", st, "!"
            call f90fflush()
            stop
         end if
         null_vector = 0
   
         CALL MGV( dx, struct, null_vector, temp_vector, level-1, v1, v2, w )

         !CALL dealloc( temp_vector, "temp_vector in MGV" )
         deallocate( temp_vector, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector temp_vector in MGV, status=", st, "!"
            call f90fflush()
         end if
   
         tsize = int(sqrt(real(size(dx))))         
         allocate( temp_matrix1(1:tsize,1:tsize), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of matrix temp_matrix1 for transform_vector_to_matrix, &
                                    & status=", st, "!"
            call f90fflush()
            stop
         end if
         temp_matrix1 = 0
         CALL transform_vector_to_matrix( temp_matrix1, dx )
         deallocate( dx, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector dx after transform_vector_to_matrix, &
                                    & status=", st, "!"
            call f90fflush()
         end if
   
         tsize = 2**level + 1
         allocate( temp_matrix2(1:tsize,1:tsize), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of matrix temp_matrix2 for prolongation, status=", st, "!"
            call f90fflush()
            stop
         end if
         temp_matrix2 = 0
         CALL prolongation( temp_matrix2, temp_matrix1, level )
         deallocate( temp_matrix1, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of matrix temp_matrix1 after prolongation, status=", st, "!"
            call f90fflush()
         end if
   
         tsize = size(temp_matrix2)
         allocate( temp_vector(1:tsize), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of the vector temp_vector for transform_matrix_to_vector, &
                                    & status=", st, "!"
            call f90fflush()
            stop
         end if
         temp_vector = 0
         CALL transform_matrix_to_vector( temp_vector, temp_matrix2 )
         deallocate( temp_matrix2, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of matrix temp_matrix2 after transform_matrix_to_vector, &
                                    & status=", st, "!"
            call f90fflush()
         end if
   
         !CALL dealloc( null_vector, "null_vector in MGV" )
         deallocate( null_vector, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector null_vector in MGV, status=", st, "!"
            call f90fflush()
         end if
   
         x = x + temp_vector
         !CALL dealloc( temp_vector, "temp_vector in MGV" )
         deallocate( temp_vector, stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector temp_vector in MGV, status=", st, "!"
            call f90fflush()
         end if
   
         CALL JacobiRelax( x, struct(level)%matrix, x, b, v2, w )
      end if
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "MGV( solution, struct, x0, b, iter ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine MGV
   
   
   subroutine JacobiRelax( xk_1, Ah, xk, b, v, w )
      ! in/out
      REAL(kind=8), dimension(:) :: xk_1
      REAL(kind=8), dimension(:,:) :: Ah
      REAL(kind=8), dimension(:) :: xk, b
      INTEGER(kind=4) :: v
      REAL(kind=8) :: w
      ! additional
      REAL(kind=8), dimension(:,:), allocatable :: D_inverse
      REAL(kind=8), dimension(:), allocatable :: temp_vector1, temp_vector2
      INTEGER(kind=4) :: i, z3, z4
      
      if( INFO .OR. INFOTIME ) then
         write(*,*) "JacobiRelax( xk_1, Ah, xk, b, v, w ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
      
      !CALL alloc( D_inverse, size(Ah(:,1)), size(Ah(1,:)), "D_inverse" )
      allocate( D_inverse(1:size(Ah(:,1)),1:size(Ah(1,:))), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of matrix D_inverse in JacobiRelax, status=", st, "!"
         call f90fflush()
         stop
      end if
      D_inverse = 0

      !CALL alloc( temp_vector1, size(Ah(:,1)), "temp_vector" )
      allocate( temp_vector1(1:size(Ah(:,1))), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the vector temp_vector1 in JacobiRelax, status=", st, "!"
         call f90fflush()
         stop
      end if
      temp_vector1 = 0

      !CALL alloc( temp_vector2, size(Ah(:,1)), "temp_vector2" )
      allocate( temp_vector2(1:size(Ah(:,1))), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the vector temp_vector2 in JacobiRelax, status=", st, "!"
         call f90fflush()
         stop
      end if
      temp_vector2 = 0

      do i=1, size(Ah(:,1)), 1
         D_inverse(i,i) = 1.0 / Ah(i,i)
      end do
   
      do i=1, v, 1
         !xk_1 = xk - w * D_inverse * ( Ah * xk - b )
         CALL MatVecMult( temp_vector1, Ah, xk )
         temp_vector1 = temp_vector1 - b
         CALL MatVecMult( temp_vector2, D_inverse, temp_vector1 )
         xk_1 = xk - w * temp_vector2
         xk = xk_1
      end do
   
      !CALL dealloc( D_inverse, "D_inverse" )
      deallocate( D_inverse, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of matrix D_inverse in JacobiRelax, status=", st, "!"
         call f90fflush()
      end if

      !CALL dealloc( temp_vector1, "temp_vector" )
      deallocate( temp_vector1, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector temp_vector1 in JacobiRelax, status=", st, "!"
         call f90fflush()
      end if

      !CALL dealloc( temp_vector2, "temp_vector2" )
      deallocate( temp_vector2, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector temp_vector2 in JacobiRelax, status=", st, "!"
         call f90fflush()
      end if
      
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "JacobiRelax( xk_1, Ah, xk, b, v, w ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine JacobiRelax
   
   
   subroutine transform_vector_to_matrix( matrix, vector )
      ! in/out
      REAL(kind=8), dimension(:,:) :: matrix
      REAL(kind=8), dimension(:) :: vector
      ! additional
      INTEGER(kind=4) :: i, n, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "transform_vector_to_matrix( matrix, vector ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
   
      n = int(sqrt(real(size(vector))))
   
      !CALL alloc( matrix, n, n, "for transform vec2mat" )
      
      !matrix = reshape( vector, (/ n, n /))
      if( INFO ) call system_clock( z1 )
      !$omp parallel do shared(matrix, vector, n) private(i)
      do i=1, n, 1
          matrix(n-i+1,:) = vector((i-1)*n+1:(i-1)*n+n)
      end do
      !$omp end parallel do
      if( INFO ) call system_clock( z2 )
      if( INFO ) write(*,*) "time8 in clockcycle: ", z2-z1
   
      !CALL dealloc( vector, "for transform vec2mat" )
      
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "transform_vector_to_matrix( matrix, vector ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine transform_vector_to_matrix
   
   
   subroutine transform_matrix_to_vector( vector, matrix )
      ! in/out
      REAL(kind=8), dimension(:) :: vector
      REAL(kind=8), dimension(:,:) :: matrix
      ! additional
      INTEGER(kind=4) :: i, n, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "transform_matrix_to_vector( vector, matrix ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
   
      !CALL alloc( vector, size(matrix), "for transform mat2vec" )
   
      n = size(matrix(:,1))
   
      !vector = reshape( matrix, (/ n /))
      if( INFO ) call system_clock( z1 )
      !$omp parallel do shared(vector, matrix, n) private(i)
      do i=1, n, 1
          vector((i-1)*n+1:(i-1)*n+n) = matrix(n-i+1,:)
      end do
      !$omp end parallel do
      if( INFO ) call system_clock( z2 )
      if( INFO ) write(*,*) "time9 in clockcycle: ", z2-z1
   
      !CALL dealloc( matrix, "for transform mat2vec" )
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "transform_matrix_to_vector( vector, matrix ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine transform_matrix_to_vector
   
   
   subroutine restriction( u, r0, level )
      ! in/out
      REAL(kind=8), dimension(:,:) :: u, r0         ! u=outMatrix, r0=inMatrix
      INTEGER(kind=4) :: level
      ! additional
      REAL(kind=8), dimension(:,:), allocatable :: r   ! r=blowUpMatrix
      INTEGER(kind=4) :: i, j, n, z3, z4
      
      if( INFO .OR. INFOTIME ) then
         write(*,*) "restriction( u, r0, level ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
      
      n = size(r0(:,1))+2
      allocate( r(1:n,1:n), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of vector r in restriction, status=", st, "!"
         call f90fflush()
         stop
      end if

      r(:,1) = 0
      r(:,n) = 0
      r(1,:) = 0
      r(n,:) = 0
      
      r(2:2**level+2, 2:2**level+2) = r0
   
      n = 2**(level-1) + 1
   
      !CALL dealloc( r0, "r0" )
      !CALL alloc( u, n, n, "u in restric" )
   
      if( INFO ) call system_clock( z1 )
      !$omp parallel do shared(u, r, n) private(j, i)
      do j=1, n, 1
         do i=1, n, 1
            u(i,j) = 1.0/4.0 * r(2*i,2*j) + 1.0/8.0 * (r(2*i,2*j-1)+r(2*i,2*j+1)+r(2*i-1,2*j)+r(2*i+1,2*j)) + &
                & 1.0/16.0 * (r(2*i+1,2*j+1)+r(2*i-1,2*j+1)+r(2*i+1,2*j-1)+r(2*i-1,2*j-1))
         end do
      end do
      !$omp end parallel do
      if( INFO ) call system_clock( z2 )
      if( INFO ) write(*,*) "time10 in clockcycle: ", z2-z1
      
      deallocate( r, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector r in restriction, status=", st, "!"
         call f90fflush()
      end if
      
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "restriction( u, r0, level ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine restriction
   
   
   subroutine prolongation( u, dx, level )
      ! in/out
      REAL(kind=8), dimension(:,:) :: u, dx         ! u=outMatrix, dx=inMatrix
      INTEGER(kind=4) :: level
      ! additional
      INTEGER(kind=4) :: i, j, n, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "prolongation( u, dx, level ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
   
      n = 2**level + 1
   
      !CALL alloc( u, n, n, "u in prolong" )
   
      u(1:n:2,1:n:2) = dx(1:2**(level-1)+1,1:2**(level-1)+1)
   
      !CALL dealloc( dx, "dx" )

      if( INFO ) call system_clock( z1 )
      !$omp parallel do shared(u, n) private(i, j)
      do i=2, n, 2
         do j=1, n, 2
            u(i,j) = 1.0/2.0 * (u(i-1,j)+u(i+1,j))
            u(j,i) = 1.0/2.0 * (u(j,i-1)+u(j,i+1))
         end do
      end do
      !$omp end parallel do
      if( INFO ) call system_clock( z2 )
      if( INFO ) write(*,*) "time11 in clockcycle: ", z2-z1
   
      if( INFO ) call system_clock( z1 )
      !$omp parallel do shared(u, n) private(j, i)
      do j=2, n, 2
         do i=2, n, 2
            u(i,j) = 1.0/4.0 * (u(i-1,j-1)+u(i-1,j+1)+u(i+1,j-1)+u(i+1,j+1))
         end do
      end do
      !$omp end parallel do
      if( INFO ) call system_clock( z2 )
      if( INFO ) write(*,*) "time12 in clockcycle: ", z2-z1
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "prolongation( u, dx, level ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine prolongation
   
   
!   subroutine removeMatrixStruct( struct )
!      ! in/out
!      type(matrixStruct), dimension(:), allocatable :: struct
!      ! additional
!      INTEGER(kind=4) :: i, st, z3, z4
!   
!      if( INFO .OR. INFOTIME ) then
!         write(*,*) "removeMatrixStruct( struct ) in"
!         call system_clock( z3 )
!         call f90fflush()
!      end if
!   
!      do i=1, size(struct), 1
!         deallocate(struct(i)%matrix, stat=st)
!         if( st/=0 ) write(*,fmt='(A,I0,A,I0,A)') "Error: Deallocation of the matrix for level=", i, &
!                                                  & " not possible, status=", st, "!"
!      end do
!   
!      deallocate(struct, stat=st)
!      if( st/=0 ) write(*,fmt='(A,I0,A)') "Error: Deallocation of the struct not possible, status=", st, "!"
!   
!      if( INFO .OR. INFOTIME ) then
!         call system_clock( z4 )
!         write(*,*) "removeMatrixStruct( struct ) out, time for function: ", z4-z3
!         call f90fflush()
!      end if
!   end subroutine removeMatrixStruct
   
   
   subroutine printMatrixStruct( struct )
      ! in/out
      type(matrixStruct), dimension(:) :: struct
      ! additional
      INTEGER(kind=4) :: i, row, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "printMatrixStruct( struct ) in"
         call system_clock( z3 )
         call f90fflush()
      end if

      do i=1, size(struct), 1
         write(*,fmt='(A,I0,A)') "Matrix of Struct[", i, "] = "
         do row=1, size(struct(i)%matrix(:,1)), 1
            write(*,*) struct(i)%matrix(row,:)
         end do
      end do

      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "printMatrixStruct( struct ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine printMatrixStruct
   
   
   subroutine writeOutput( outputform, level0, solution, L1, L2 )
      ! in/out
      INTEGER(kind=4) :: outputform, level0
      REAL(kind=8), dimension(:) :: solution
      REAL(kind=8) :: L1, L2
      ! additional
      REAL(kind=8), dimension(:), allocatable :: x1, x2
      INTEGER(kind=4) :: i, j, N1, st, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "writeOutput( outputform, level0, solution, L1, L2 ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
   
      N1 = 2**level0+1
      
      !CALL alloc( x1, N1, "x1 in writeOutput" )
      allocate( x1(1:N1), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the vector x1 in writeOutput, status=", st, "!"
         call f90fflush()
         stop
      end if
      x1 = 0

      !CALL alloc( x2, N1, "x2 in writeOutput" )
      allocate( x2(1:N1), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the vector x2 in writeOutput, status=", st, "!"
         call f90fflush()
         stop
      end if
      x2 = 0
   
      CALL get_x1( x1, N1, L1)
      CALL get_x2( x2, N1, L2)
   
      if( outputform==1 ) then
         open( unit=100, file="output.xyz", iostat=st, status='replace', position='rewind', action='write' )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Opening file output.xyz, status=", st, "!"
            call f90fflush()
         end if
         open( unit=200, file="output.plt", iostat=st, status='replace', position='rewind', action='write' )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Opening file output.plt, status=", st, "!"
            call f90fflush()
         end if
   
         ! write output.xyz
         do j=1, size(x2), 1
            do i=1, size(x1), 1
               write(100,fmt='(E21.15,A,E21.15,A,E21.15)') x1(i), " ", x2(j), " ", solution((j-1)*N1+i)
            end do
         end do
   
         ! write output.plt version 1
         !write(200,fmt='(A)') "set pm3d"
         !write(200,fmt='(A,I0,A,I0,A)') "set dgrid3d ",N1,", ",N1,", 1"
         !write(200,fmt='(A)') "splot 'output.xyz' with pm3d"
         !write(200,fmt='(A)') "pause -1"
   
         ! write output.plt version 2
         write(200,fmt='(A)') "set pm3d at b"
         write(200,fmt='(A,I0,A,I0,A)') "set dgrid3d ",N1,", ",N1,", 1"
         write(200,fmt='(A)') "splot 'output.xyz' with lines"
         write(200,fmt='(A)') "pause -1"
   
         close( unit=100, iostat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Closing file output.xyz, status=", st, "!"
            call f90fflush()
         end if
         close( unit=200, iostat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Closing file output.plt, status=", st, "!"
            call f90fflush()
         end if
      end if

      !CALL dealloc( x1, "x1 in writeOutput" )
      deallocate( x1, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector x1 in writeOutput, status=", st, "!"
         call f90fflush()
      end if

      !CALL dealloc( x2, "x2 in writeOutput" )
      deallocate( x2, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of the vector x2 in writeOutput, status=", st, "!"
         call f90fflush()
      end if
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "writeOutput( outputform, level0, solution, L1, L2 ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine writeOutput


   function calcFlop(level0, counter, v1, v2)
      ! in/out
      INTEGER(kind=4) :: level0, counter, v1, v2
      REAL(kind=8) :: calcFlop
      ! additional
      INTEGER(kind=4) :: i
      REAL(kind=8) flop, N

      flop = 0
      
      ! flop in my2dPoisson_without_sparse
      N = real( (2**level0+1)**2 )	! size of a vector, example solution
      flop = flop + 2*N*N	! MatVecMult
      flop = flop + N		! b-r
      flop = flop + N + 2	! abs and res2
      
      ! MGV (level/=1)
      N = real( (2**1+1)**2 )
      flop = flop + 2*N*N

      ! MGV (level/=1)
      do i=level0, 2, -1
         ! jacobi
         N = real( (2**i+1)**2 )
         flop = flop + N					! D_inv
         flop = flop + v1 * ( 2*N*N + N + 2*N*N + N + N )	! second loop
         
         flop = flop + 2*N*N	! MatVecMult
         flop = flop + N	! b-r

         ! restriction
         !r0 -> N, level -> i, n -> 2**(i-1) + 1
         flop = flop + 14 * real((2**(i-1)+1)) * real((2**(i-1)+1))

         ! prolongation
         !dx -> (2**(i-1)+1)**2, level -> i, n -> 2**i + 1
         flop = flop + 6 * (real(2**i+1)/2) * (real(2**i+1)/2 + 1)
         flop = flop + 5 * (real(2**i+1)/2) * (real(2**i+1)/2)

         flop = flop + N	! x+temp_vec

         ! jacobi
         flop = flop + N					! D_inv
         flop = flop + v2 * ( 2*N*N + N + 2*N*N + N + N )	! second loop
      end do

      calcFlop = counter * flop
   end function calcFlop


   subroutine MatVecMult( out_vector, matrix, vector )
      ! in/out
      REAL(kind=8), dimension(:) :: out_vector
      REAL(kind=8), dimension(:,:) :: matrix
      REAL(kind=8), dimension(:) :: vector
      ! additional
      INTEGER(kind=4) :: i, j, m, n, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "MatVecMult( out_vector, matrix, vector ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
      
      out_vector = 0

      if( INFO ) call system_clock( z1 )
      m = size(matrix(:,1))
      n = size(matrix(1,:))
      !$omp parallel do shared(out_vector, matrix, vector, m, n) private(i, j)
      do i=1, m, 1
         do j=1, n, 1
            out_vector(i) = out_vector(i) + matrix(i,j) * vector(j)
         end do
      end do
      !$omp end parallel do
      if( INFO ) call system_clock( z2 )
      if( INFO ) write(*,*) "time13 in clockcycle: ", z2-z1

      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "MatVecMult( out_vector, matrix, vector ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine MatVecMult

end module my2dPoisson

!************ main programm routines ************


!************ main programm entry ************

subroutine fortran_entry(  level0, maxlevel0, outputform, v1, v2, &
                                       & w, L1, L2, time_for_MGV, omega, flop  )
   use my2dPoisson

   ! in/out
   INTEGER(kind=4) :: level0, maxlevel0, outputform, v1, v2
   REAL(kind=8) :: w, L1, L2
   REAL(kind=8) :: time_for_MGV, omega, flop

   CALL my2dPoisson_without_sparse( level0, maxlevel0, outputform, v1, v2, w, L1, L2, time_for_MGV, omega, flop )
end subroutine fortran_entry

!************ main programm entry ************



