!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! BenchIT - Performance Measurement for Scientific Applications
! Contact: developer@benchit.org
!
! $Id: 2dPoisson.f90 1 2009-09-11 12:26:19Z william $
! $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/applications/MGV/F95/0/0/no_matrix/2dPoisson.f90 $
! For license details see COPYING in the package base directory
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Kernel: multigrid methode for 2D Poisson equation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module debug
   implicit none

   LOGICAL, parameter :: INFO=.FALSE., INFOTIME=.FALSE.     ! is used for debug output
   INTEGER(kind=4) :: st, st2, st3

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

   implicit none

   type crs
      REAL(kind=8), dimension(:), allocatable :: values
      INTEGER(kind=4), dimension(:), allocatable :: colInd 
      INTEGER(kind=4), dimension(:), allocatable :: crPtr
   end type crs

   contains

   subroutine my2dPoisson_without_sparse( level0, maxlevel0, outputform, v1, v2, &
                                          & w, L1, L2, time_for_MGV, omega, flop )
      ! in/out
      INTEGER(kind=4) :: level0, maxlevel0, outputform, v1, v2
      REAL(kind=8) :: w, L1, L2
      REAL(kind=8) :: time_for_MGV, omega, flop
      ! additional
      INTEGER(kind=4) :: iter, start, counter, i, M1M1, numnz, z3, z4
      REAL(kind=8) :: res, res2, old_res
      REAL(kind=8) :: t1, t2
      CHARACTER(len=30) :: str

      REAL(kind=8), dimension(:), allocatable :: b, solution, r, x0
      REAL(kind=8), dimension(:,:), allocatable :: temp_matrix
      
      if( INFO .OR. INFOTIME ) then
         write(*,*) "my2dPoisson_without_sparse( level0, maxlevel0, outputform, &
                    & v1, v2, w, L1, L2, time_for_MGV, omega, flop ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
   
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
   
      allocate( temp_matrix(1:2**level0+1,1:2**level0+1), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of matrix temp_matrix for simulatedMultA, &
                                 & status=", st, "!"
         call f90fflush()
         stop
      end if
      temp_matrix = 0

      CALL bi_gettime( t1 )
      do
         CALL MGV( solution, x0, b, level0, v1, v2, w )
         ! r = b'-detr_vec( mulA( tr_vec(loesung), level ) )'
         
         CALL transform_vector_to_matrix( temp_matrix, solution )
         CALL simulatedMultA( temp_matrix, level0 )
         CALL transform_matrix_to_vector( r, temp_matrix )
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
   
      deallocate( temp_matrix, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of matrix temp_matrix after simulatedMultA, status=", st, "!"
         call f90fflush()
      end if

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
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "my2dPoisson_without_sparse( level0, maxlevel0, outputform, v1, v2, & 
                    & w, L1, L2, time_for_MGV, omega, flop ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine my2dPoisson_without_sparse
   
   
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
   
         do j=1, N1, 1
             do i=1, N1, 1
               !b((j-1)*((N1-1)+1)+i) = 2 * ( x1(i)*(1-(x1(i))) + x2(j)*(1-(x2(j))) )
               b((j-1)*N1+i) = 2 * ( x1(i)*(1-(x1(i))) + x2(j)*(1-(x2(j))) )
             end do
         end do
   
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
   
      do i=1, N1, 1
         x1(i) = ((i-1) * L1) / (N1-1)
      end do
   
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
   
      do i=1, N1, 1
         x2(i) = ((i-1) * L2) / (N1-1)
      end do
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "get_x2( x2, N1, L2 ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine get_x2
   
   
   recursive subroutine MGV( x, x0, b, level, v1, v2, w )
      ! in/out
      REAL(kind=8), dimension(:) :: x
      REAL(kind=8), dimension(:) :: x0, b
      INTEGER(kind=4) :: level, v1, v2
      REAL(kind=8) :: w
      ! additional
      INTEGER(kind=4) :: row, n, n1, n1n1, tsize, z3, z4
      REAL(kind=8), dimension(:), allocatable :: null_vector, r, temp_vector, dx
      REAL(kind=8), dimension(:,:), allocatable :: temp_matrix1, temp_matrix2, inverse

      if( INFO .OR. INFOTIME ) then
         write(*,*) "MGV( x, x0, b, level, v1, v2, w ) in"
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
         CALL JacobiRelax( x, x0, b, v1, w, level )

         !r = b - struct(level)%matrix * x
         !CALL alloc( r, size(x), "r in MGV" )
         allocate( r(1:size(x)), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of the vector r in MGV, status=", st, "!"
            call f90fflush()
            stop
         end if
         r = 0

         tsize = int(sqrt(real(size(r))))         
         allocate( temp_matrix1(1:tsize,1:tsize), stat=st )
         if( st/=0 ) then
            write(*,fmt='(A,I0,A)') "Error: Allocation of matrix temp_matrix1 for simulatedMultA, &
                                    & status=", st, "!"
            call f90fflush()
            stop
         end if
         temp_matrix1 = 0

         CALL transform_vector_to_matrix( temp_matrix1, x )
         CALL simulatedMultA( temp_matrix1, level )
         CALL transform_matrix_to_vector( r, temp_matrix1 )
         r = b - r

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
   
         CALL MGV( dx, null_vector, temp_vector, level-1, v1, v2, w )

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
   
         CALL JacobiRelax( x, x, b, v2, w, level )
      end if
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "MGV( x, x0, b, level, v1, v2, w ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine MGV
   
   
   subroutine JacobiRelax( xk_1, xk, b, v, w, level )
      ! in/out
      REAL(kind=8), dimension(:) :: xk_1
      type(crs) :: Ah
      REAL(kind=8), dimension(:) :: xk, b
      INTEGER(kind=4) :: v, level
      REAL(kind=8) :: w, h, h2
      ! additional
      type(crs) :: D_inverse
      REAL(kind=8), dimension(:), allocatable :: temp_vector1, temp_vector2
      REAL(kind=8), dimension(:,:), allocatable :: temp_matrix
      INTEGER(kind=4) :: i, j, nn, n, z3, z4
      
      if( INFO .OR. INFOTIME ) then
         write(*,*) "JacobiRelax( xk_1, xk, b, v, w, level ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
      
      h = 1/real(2**level)
      h2 = 0.25*(h**2)
      nn = size(xk)
      n = int(sqrt(real(nn)))
      
      !CALL alloc( D_inverse, size(Ah(:,1)), size(Ah(1,:)), "D_inverse" )
      allocate( D_inverse%values(1:nn), stat=st )
      allocate( D_inverse%colInd(1:nn), stat=st2 )
      allocate( D_inverse%crPtr(1:nn+1), stat=st3 )
      if( st/=0 .or. st2/=0 .or. st3/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of matrix D_inverse in JacobiRelax, status=", st, st2, st3, "!"
         call f90fflush()
         stop
      end if
      D_inverse%values = 0
      D_inverse%colInd = 0
      D_inverse%crPtr = 0

      !CALL alloc( temp_vector1, size(Ah(:,1)), "temp_vector" )
      allocate( temp_vector1(1:nn), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the vector temp_vector1 in JacobiRelax, status=", st, "!"
         call f90fflush()
         stop
      end if
      temp_vector1 = 0

      !CALL alloc( temp_vector2, size(Ah(:,1)), "temp_vector2" )
      allocate( temp_vector2(1:nn), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of the vector temp_vector2 in JacobiRelax, status=", st, "!"
         call f90fflush()
         stop
      end if
      temp_vector2 = 0

      ! generate inverse of the main diagonal of matrix Ah 
      D_inverse%crPtr(1) = 1
      do i=2, nn+1, 1
         D_inverse%crPtr(i) = D_inverse%crPtr(i-1) + 1
      end do
      D_inverse%colInd(1) = 1
      do i=2, nn, 1
         D_inverse%colInd(i) = D_inverse%colInd(i-1) + 1
      end do

      D_inverse%values = -w * h2
      do i=1, n, 1
         D_inverse%values(i) = -w
      end do
      do i=nn-(n-1), nn, 1
         D_inverse%values(i) = -w
      end do
      do i=n+1, nn, n
         D_inverse%values(i-1:i) = (/-w, -w/)
      end do

      allocate( temp_matrix(1:n,1:n), stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Allocation of matrix temp_matrix for JacobiRelax, &
                                 & status=", st, "!"
         call f90fflush()
         stop
      end if
      temp_matrix = 0

      xk_1 = xk
      do i=1, v, 1
         ! xk_1 = xk_1 + D_inv*(detr_vec(mulA(tr_vec(xk_1), level))' - b)
         CALL transform_vector_to_matrix( temp_matrix, xk_1 )
         CALL simulatedMultA( temp_matrix, level )
         CALL transform_matrix_to_vector( temp_vector1, temp_matrix )
         temp_vector1 = temp_vector1 - b
         CALL MatVecMult_sparse( temp_vector2, D_inverse%values, D_inverse%colInd, D_inverse%crPtr, temp_vector1 )
         xk_1 = xk_1 + temp_vector2
      end do
   
      !CALL dealloc( D_inverse, "D_inverse" )
      deallocate( D_inverse%values, stat=st )
      deallocate( D_inverse%colInd, stat=st2 )
      deallocate( D_inverse%crPtr, stat=st3 )
      if( st/=0 .or. st2/=0 .or. st3/=0) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of matrix D_inverse in JacobiRelax, status=", st, st2, st3, "!"
         call f90fflush()
      end if

      deallocate( temp_matrix, stat=st )
      if( st/=0 ) then
         write(*,fmt='(A,I0,A)') "Error: Deallocation of matrix temp_matrix after simulatedMultA, status=", st, "!"
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
         write(*,*) "JacobiRelax( xk_1, xk, b, v, w, level ) out, time for function: ", z4-z3
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
      do i=1, n, 1
          matrix(n-i+1,:) = vector((i-1)*n+1:(i-1)*n+n)
      end do
   
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
      do i=1, n, 1
          vector((i-1)*n+1:(i-1)*n+n) = matrix(n-i+1,:)
      end do
   
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
   
      do j=1, n, 1
         do i=1, n, 1
            u(i,j) = 1.0/4.0 * r(2*i,2*j) + 1.0/8.0 * (r(2*i,2*j-1)+r(2*i,2*j+1)+r(2*i-1,2*j)+r(2*i+1,2*j)) + &
                & 1.0/16.0 * (r(2*i+1,2*j+1)+r(2*i-1,2*j+1)+r(2*i+1,2*j-1)+r(2*i-1,2*j-1))
         end do
      end do
      
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
   
      do i=2, n, 2
         do j=1, n, 2
            u(i,j) = 1.0/2.0 * (u(i-1,j)+u(i+1,j))
            u(j,i) = 1.0/2.0 * (u(j,i-1)+u(j,i+1))
         end do
      end do
      do j=2, n, 2
         do i=2, n, 2
            u(i,j) = 1.0/4.0 * (u(i-1,j-1)+u(i-1,j+1)+u(i+1,j-1)+u(i+1,j+1))
         end do
      end do
   
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "prolongation( u, dx, level ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine prolongation
   
   
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
      N = real( (2**level0+1)**2 )		! size of a vector, example solution
      flop = flop + 4
      flop = flop + 5*N - 5*(4*sqrt(N))	! simulatedMultA
      flop = flop + N					! b-r
      flop = flop + N + 2				! abs and res2
      
      ! MGV (level/=1)
      N = real( (2**1+1)**2 )
      flop = flop + 2*N*N

      ! MGV (level/=1)
      do i=level0, 2, -1
         ! jacobi
         N = real( (2**i+1)**2 )
         flop = flop + 5
         flop = flop + N					! D_inv
         flop = flop + v1 * ( 5*N - 5*(4*sqrt(N)) + N + 2*( 5 * N - 4 * ( 4*(2**i + 1) - 4 ) ) + N )	! second loop
         
         flop = flop + 5*N - 5*(4*sqrt(N))	! MatVecMult
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
         flop = flop + v2 * ( 5*N - 5*(4*sqrt(N)) + N + 2*( 5 * N - 4 * ( 4*(2**i + 1) - 4 ) ) + N )	! second loop
      end do

      calcFlop = counter * flop
   end function calcFlop


   subroutine MatVecMult( out_vector, matrix, vector )
      ! in/out
      REAL(kind=8), dimension(:) :: out_vector
      REAL(kind=8), dimension(:,:) :: matrix
      REAL(kind=8), dimension(:) :: vector
      ! additional
      INTEGER(kind=4) :: i, j, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "MatVecMult( out_vector, matrix, vector ) in"
         call system_clock( z3 )
         call f90fflush()
      end if
      
      out_vector = 0

      do i=1, size(matrix(1,:)), 1
         do j=1, size(matrix(:,1)), 1
            out_vector(j) = out_vector(j) + matrix(j,i) * vector(i)
         end do
      end do
      
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "MatVecMult( out_vector, matrix, vector ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine MatVecMult


   subroutine MatVecMult_sparse( out_vector, values, colInd, crPtr, vector )
      ! in/out
      REAL(kind=8), dimension(:) :: out_vector
      ! type(crs) :: sparse
      REAL(kind=8), dimension(:) :: values
      INTEGER(kind=4), dimension(:) :: colInd, crPtr
      REAL(kind=8), dimension(:) :: vector
      ! additional
      INTEGER(kind=4) :: i, j, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "MatVecMult_sparse( out_vector, values, colInd, crPtr, vector ) in"
         call system_clock( z3 )
         call f90fflush()
      end if

      out_vector = 0
      
      do j=1, size(crPtr)-1, 1
         do i=crPtr(j), crPtr(j+1)-1, 1
            out_vector(j) = out_vector(j) + values(i) * vector( colInd(i) )
         end do
      end do
      
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "MatVecMult_sparse( out_vector, values, colInd, crPtr, vector ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine MatVecMult_sparse


   subroutine simulatedMultA( u, level )
      ! in/out
      REAL(kind=8), dimension(:,:) :: u
      INTEGER(kind=4) :: level
      ! additional
      REAL(kind=8), dimension(1:size(u,1),1:size(u,2)) :: u_out
      REAL(kind=8) :: h, h2
      INTEGER(kind=4) :: i, j, z3, z4
   
      if( INFO .OR. INFOTIME ) then
         write(*,*) "simulatedMultA( u, level ) in"
         call system_clock( z3 )
         call f90fflush()
      end if

      h = 1/real(2**level)
      h2 = h**2
      h2 = 1/h2

      u_out(1,:) = u(1,:)
      u_out(size(u,1),:) = u(size(u,1),:)
      u_out(:,1) = u(:,1)
      u_out(:,size(u,2)) = u(:,size(u,2))

      do i=2, size(u,1)-1, 1
         do j=2, size(u,2)-1, 1
            u_out(i,j) = h2*((-u(i,j-1) + 4*u(i,j) - u(i,j+1)) - (u(i-1,j) + u(i+1,j)))
         end do
      end do
      u = u_out
      
      if( INFO .OR. INFOTIME ) then
         call system_clock( z4 )
         write(*,*) "simulatedMultA( u, level ) out, time for function: ", z4-z3
         call f90fflush()
      end if
   end subroutine simulatedMultA

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


