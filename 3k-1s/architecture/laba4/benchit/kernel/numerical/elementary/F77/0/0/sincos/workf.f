CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C BenchIT - Performance Measurement for Scientific Applications
C Contact: developer@benchit.org
C
C $Id: work.f 1 2009-09-11 12:26:19Z william $
C $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/elementary/F77/0/0/sincos/work.f $
C For license details see COPYING in the package base directory
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Kernel: Execution time of mathematical operations SINUS / COSINUS
C         addict to input value
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE mathopsin( x, y)
          double precision x, y
          if(x .NE. 0.0) y = sin(x)
      END

      SUBROUTINE mathopcos( x, y)
         double precision x, y
         if(x .NE. 0.0) y = cos(x)
      END

