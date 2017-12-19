CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C BenchIT - Performance Measurement for Scientific Applications
C Contact: developer@benchit.org
C
C $Id: cleancache.f 1 2009-09-11 12:26:19Z william $
C $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/dotproduct/F77/0/0/double/cleancache.f $
C For license details see COPYING in the package base directory
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Kernel: Core for dot product of two vectors
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      subroutine cleancache(x,n)
         double precision x(n)
         double precision s
         call init(x(1),x(n/2+1),s,n/2)
         return
      end

