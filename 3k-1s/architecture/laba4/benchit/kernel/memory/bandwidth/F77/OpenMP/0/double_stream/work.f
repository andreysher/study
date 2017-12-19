CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C BenchIT - Performance Measurement for Scientific Applications
C Contact: developer@benchit.org
C
C $Id: work.f 1 2009-09-11 12:26:19Z william $
C $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/bandwidth/F77/OpenMP/0/double_stream/work.f $
C For license details see COPYING in the package base directory
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Kernel: measure Bandwidth inspired by STREAM benchmark
C         (FORTRAN OMP-version)
C
C         according to the rules, reffer this Benchmark as:
C         "BenchIT kernel based on a variant of the STREAM benchmark code"
C         when publishing results
C
C         This file contains the work, that is done: copy,scale,add
C         and triad
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE copy( c ,a , n)
	INTEGER*4 n,i
	double precision a(n)
	double precision c(n)
!$omp parallel do
	do 100 i=1,n
	   c(i)=a(i)
 100	CONTINUE
	RETURN
	END
	
	SUBROUTINE scale( b, c, scalar, n)
	INTEGER*4 n,i
	double precision b(n)
	double precision c(n)
	double precision scalar
!$omp parallel do
	do 100 i=1,n
	   b(i)=scalar*c(i)
 100	CONTINUE
	RETURN
	END
	
	SUBROUTINE add( c, a, b, n)
	INTEGER*4 n,i
	double precision a(n)
	double precision b(n)
	double precision c(n)
!$omp parallel do
	do 100 i=1,n
	   c(i)=a(i)+b(i)
 100	CONTINUE
	RETURN
	END
	
	SUBROUTINE triad( a, b, c, scalar, n)
	INTEGER*4 n,i
	double precision a(n)
	double precision b(n)
	double precision c(n)
	double precision scalar
!$omp parallel do
	do 100 i=1,n
	   a(i)=b(i)+scalar*c(i)
 100	CONTINUE
	RETURN
	END
