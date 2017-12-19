!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! BenchIT - Performance Measurement for Scientific Applications
! Contact: developer@benchit.org
!
! $Id: gauss.f90 1 2009-09-11 12:26:19Z william $
! $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/gauss/F95/0/0/double/gauss.f90 $
! For license details see COPYING in the package base directory
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Kernel: Gaussian Linear Equation System Solver
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! seriell Gauss
subroutine entry(A,b,x,n)
   implicit none

   integer(kind=4) :: i, i1, i2, j0, j1, j2, j, k, n, m
   real(kind=8)    :: A(n,n), mi, b(n), t, x(n), bi, mt, t1
!****************************************************************
   do i=1,n-1,1
      t=1.0D0/A(i,i)
      do j=i+1,n
         b(j)=b(j)-b(i)*(A(j,i)*t)
         do k=i+1,n
            A(k,j)=A(k,j)-(A(i,j)*t)*A(k,i)
         end do
      end do
   end do

   x(n)=b(n)/A(n,n)
   do i=n-1,1,-1
      x(i)=b(i)
      do j=i+1,n
         x(i)=x(i)-A(i,j)*x(j)
      end do
      x(i)=x(i)/A(i,i)
   end do
!****************************************************************
end subroutine

