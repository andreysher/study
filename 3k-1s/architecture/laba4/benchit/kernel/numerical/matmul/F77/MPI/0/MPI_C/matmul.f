CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C BenchIT - Performance Measurement for Scientific Applications
C Contact: developer@benchit.org
C
C $Id: matmul.f 1 2009-09-11 12:26:19Z william $
C $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/F77/MPI/0/MPI_C/matmul.f $
C For license details see COPYING in the package base directory
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Kernel: Matrix multiply (Fortran 77) with MPI
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      subroutine tstcas( n, m, itype, a, b, c)
      Integer n, m, itype
      Real*8 a( n, n ), b( n, n ), c( n, n ), temp

      Integer i11 ,j1 ,j2 , i, j

C     **** Initialization  
      i11 = 0
      j1 = 0
      j2 = 0
      i = 0
      j = 0

      if ( itype.eq.1 ) then 
         Do i11 = 1, m
            do j = 1, n
               do i = 1, n
                  do k = 1, n
                     c( i, j ) = c( i, j ) + a( i, k ) * b( k ,j )
                  enddo
               end do
            end do
         enddo   
      else
         if ( itype.eq.2 ) then 
            do i11 = 1, m
               do j = 1, n
                  do i = 1, n
                     temp = c( i, j )
                     do k = 1, n
                        temp = temp + a( i, k ) * b( k, j )
                     enddo
                     c( i, j ) = temp
                  enddo  
               end do
            end do
         elseif ( itype.eq.3 ) then 
            do i11 = 1, m
               do j = 1, n
                  do k = 1, n
                     do i = 1, n
                        c( i, j ) = c( i, j ) + a( i, k ) * b( k, j )
                     enddo
                  enddo  
               end do
            end do
         elseif ( itype.eq.4 ) then 
            do i11 = 1, m
               do k = 1, n
                  do j = 1, n
                     do i = 1, n
                        c( i, j ) = c( i, j ) + a( i, k ) * b( k, j )
                     enddo
                  enddo  
               end do
            end do
         elseif ( itype.eq.5 ) then 
            do i11 = 1, m
               do k = 1, n
                  do i = 1, n
                     do j = 1, n
                        c( i, j ) = c( i, j ) + a( i, k ) * b( k, j )
                     enddo
                  enddo  
               end do
            end do
         elseif ( itype.eq.6 ) then 
            do i11 = 1, m
               do i = 1, n
                  do k = 1, n
                     do j = 1, n
                        c( i, j ) = c( i, j ) + a( i, k ) * b( k, j )
                     enddo
                  enddo  
               end do
            end do
         elseif ( itype.eq.7 ) then 
            do i11 = 1, m
               do i = 1, n
                  do j = 1, n
                     do k = 1, n
                        c( i, j ) = c( i, j ) + a( i, k ) * b( k, j )
                     enddo
                  enddo  
               end do
            end do
         endif
      endif

C      sum = 0
C      Do j=1,n
C         do i=1,n
C            sum = sum + c(i,j)
C         enddo
C      enddo
      end
