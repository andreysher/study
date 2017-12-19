CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C BenchIT - Performance Measurement for Scientific Applications
C Contact: developer@benchit.org
C
C $Id: dotproduct.f 1 2009-09-11 12:26:19Z william $
C $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/dotproduct/F77/0/0/double/dotproduct.f $
C For license details see COPYING in the package base directory
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Kernel: Core for dot product of two vectors
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      subroutine dotproduct(x,y,n,s,numthreads,dynamic,cacheflush,
     &                      minl,maxl,flopmin,flopmax,acache,ncache)
         implicit none
         integer n,s,numthreads,dynamic,cacheflush,ncache
         double precision x(n),y(n),minl,maxl,flopmin,
     &                    flopmax,acache(ncache)
         integer i,m
         double precision summe,startl,endl
         integer OFF
         parameter (OFF=0)
         if(s.gt.n)GOTO 10
         minl=1.0d+30
         maxl=0
         do m=1,5
            summe=0
            if(cacheflush.NE.OFF)call cleancache(acache,ncache)
            call bigtime(startl)
            do i=1,s
               summe=summe+x(i)*y(i)
            end do
            call bigtime(endl)
            minl=min(minl,endl-startl)
            maxl=max(maxl,endl-startl)
            if(summe.NE.s)stop
         end do
         if(summe.NE.s)GOTO 10
         if(maxl.EQ.0.OR.minl.EQ.0)GOTO 10
         flopmin=2*s/maxl
         flopmax=2*s/minl
         return
10       minl=0
         maxl=0
         flopmin=0.0D0
         flopmax=0.0D0
         return
      end

