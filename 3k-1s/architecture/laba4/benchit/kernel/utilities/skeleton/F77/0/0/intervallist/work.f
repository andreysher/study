CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C BenchIT - Performance Measurement for Scientific Applications
C Contact: developer@benchit.org
C
C $Id: simple.f 1 2009-09-11 12:26:19Z william $
C $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/utilities/skeleton/F77/0/0/simple/simple.f $
C For license details see COPYING in the package base directory
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Kernel: Fortran-Skeleton
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE work_1(n)
	INTEGER*4 n
	do 100 i=1,n
	   WRITE (*,*) "Hello world1!"
 100	CONTINUE
	RETURN
	END

	SUBROUTINE work_2(n)
	INTEGER*4 n
	do 100 i=1,n
	   WRITE (*,*) "Hello world2!"
 100	CONTINUE
	RETURN
	END
