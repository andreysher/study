CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C BenchIT - Performance Measurement for Scientific Applications
C Contact: developer@benchit.org
C
C For license details see COPYING in the package base directory
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C Kernel: Matrix Multiply (F77)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	SUBROUTINE multaijk( a, b, c, n)
	INTEGER*4 n,i,j,k
	double precision a(n,n), b(n,n), c(n,n)
!$omp parallel do shared(a,b,c,n) private(i,j,k)
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
	double precision a(n,n), b(n,n), c(n,n)
!$omp parallel do shared(a,b,c,n) private(i,j,k)
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
	double precision a(n,n), b(n,n), c(n,n)
!$omp parallel do shared(a,b,c,n) private(i,j,k)
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
	double precision a(n,n), b(n,n), c(n,n)
!$omp parallel do shared(a,b,c,n) private(i,j,k)
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
	double precision a(n,n), b(n,n), c(n,n)
!$omp parallel do shared(a,b,c,n) private(i,j,k)
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
	double precision a(n,n), b(n,n), c(n,n)
!$omp parallel do shared(a,b,c,n) private(i,j,k)
	do 100 k=1,n
	do 100 j=1,n
	do 100 i=1,n
	   c(i,j)=c(i,j)+a(i,k)*b(k,j)
 100	CONTINUE
	RETURN
	END
