CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C BenchIT - Performance Measurement for Scientific Applications
C Contact: developer@benchit.org
C
C $Id: matmul_f_core.f 1 2009-09-11 12:26:19Z william $
C $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/F77/0/0/integer/matmul_f_core.f $
C For license details see COPYING in the package base directory
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Kernel: Matrix Multiply (F77)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


	SUBROUTINE multaijk( a, b, c, n)
	INTEGER*4 n,i,j,k
	INTEGER*4 a(n,n), b(n,n), c(n,n)
	do 100 i=1,n
	do 100 j=1,n
	do 100 k=1,n
	   c(i,j)=c(i,j)+a(i,k)*b(k,j)
 100	CONTINUE
	RETURN
	END
c
	SUBROUTINE multaikj( a, b, c, n)
	INTEGER*4 n,i,j,k
	INTEGER*4 a(n,n), b(n,n), c(n,n)
	do 100 i=1,n
	do 100 k=1,n
	do 100 j=1,n
	   c(i,j)=c(i,j)+a(i,k)*b(k,j)
 100	CONTINUE
	RETURN
	END
c
	SUBROUTINE multajik( a, b, c, n)
	INTEGER*4 n,i,j,k
	INTEGER*4 a(n,n), b(n,n), c(n,n)
	do 100 j=1,n
	do 100 i=1,n
	do 100 k=1,n
	   c(i,j)=c(i,j)+a(i,k)*b(k,j)
 100	CONTINUE
	RETURN
	END
c
	SUBROUTINE multajki( a, b, c, n)
	INTEGER*4 n,i,j,k
	INTEGER*4 a(n,n), b(n,n), c(n,n)
	do 100 j=1,n
	do 100 k=1,n
	do 100 i=1,n
	   c(i,j)=c(i,j)+a(i,k)*b(k,j)
 100	CONTINUE
	RETURN
	END
c
	SUBROUTINE multakij( a, b, c, n)
	INTEGER*4 n,i,j,k
	INTEGER*4 a(n,n), b(n,n), c(n,n)
	do 100 k=1,n
	do 100 i=1,n
	do 100 j=1,n
	   c(i,j)=c(i,j)+a(i,k)*b(k,j)
 100	CONTINUE
	RETURN
	END
c
	SUBROUTINE multakji( a, b, c, n)
	INTEGER*4 n,i,j,k
	INTEGER*4 a(n,n), b(n,n), c(n,n)
	do 100 k=1,n
	do 100 j=1,n
	do 100 i=1,n
	   c(i,j)=c(i,j)+a(i,k)*b(k,j)
 100	CONTINUE
	RETURN
	END
	

