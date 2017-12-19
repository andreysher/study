!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! BenchIT - Performance Measurement for Scientific Applications
! Contact: developer@benchit.org
!
! $Id: matmul.f90 1 2009-09-11 12:26:19Z william $
! $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/cannon/F95/MPI/0/double/matmul.f90 $
! For license details see COPYING in the package base directory
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Kernel: a MPI version of matrix-matrix multiplication
!         (cannon algotithm)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SUBROUTINE multijk( matrixC, matrixA, matrixB, mA, nA, nB )
	INTEGER(kind=4) :: i, j, k, mA, nA, nB
	REAL(kind=8), DIMENSION(:, :) :: matrixC(mA,nB), matrixA(mA,nA), matrixB(nA,nB)

	DO i=1, mA
		DO j=1, nB
			DO k=1, nA
				matrixC(i,j) = matrixC(i,j) + matrixA(i,k)*matrixB(k,j)
			END DO
		END DO
	END DO

END SUBROUTINE multijk

SUBROUTINE multikj( matrixC, matrixA, matrixB, mA, nA, nB )
	INTEGER(kind=4) :: i, j, k, mA, nA, nB
	REAL(kind=8), DIMENSION(:, :) :: matrixC(mA,nB), matrixA(mA,nA), matrixB(nA,nB)

	DO i=1, mA
		DO k=1, nA
			DO j=1, nB
				matrixC(i,j) = matrixC(i,j) + matrixA(i,k)*matrixB(k,j)
			END DO
		END DO
	END DO

END SUBROUTINE multikj

SUBROUTINE multjik( matrixC, matrixA, matrixB, mA, nA, nB )
	INTEGER(kind=4) :: i, j, k, mA, nA, nB
	REAL(kind=8), DIMENSION(:, :) :: matrixC(mA,nB), matrixA(mA,nA), matrixB(nA,nB)

	DO j=1, nB
		DO i=1, mA
			DO k=1, nA
				matrixC(i,j) = matrixC(i,j) + matrixA(i,k)*matrixB(k,j)
			END DO
		END DO
	END DO

END SUBROUTINE multjik

SUBROUTINE multjki( matrixC, matrixA, matrixB, mA, nA, nB )
	INTEGER(kind=4) :: i, j, k, mA, nA, nB
	REAL(kind=8), DIMENSION(:, :) :: matrixC(mA,nB), matrixA(mA,nA), matrixB(nA,nB)

	DO j=1, nB
		DO k=1, nA
			DO i=1, mA
				matrixC(i,j) = matrixC(i,j) + matrixA(i,k)*matrixB(k,j)
			END DO
		END DO
	END DO

END SUBROUTINE multjki

SUBROUTINE multkij( matrixC, matrixA, matrixB, mA, nA, nB )
	INTEGER(kind=4) :: i, j, k, mA, nA, nB
	REAL(kind=8), DIMENSION(:, :) :: matrixC(mA,nB), matrixA(mA,nA), matrixB(nA,nB)

	DO k=1, nA
		DO i=1, mA
			DO j=1, nB
				matrixC(i,j) = matrixC(i,j) + matrixA(i,k)*matrixB(k,j)
			END DO
		END DO
	END DO

END SUBROUTINE multkij

SUBROUTINE multkji( matrixC, matrixA, matrixB, mA, nA, nB )
	INTEGER(kind=4) :: i, j, k, mA, nA, nB
	REAL(kind=8), DIMENSION(:, :) :: matrixC(mA,nB), matrixA(mA,nA), matrixB(nA,nB)

	DO k=1, nA
		DO j=1, nB
			DO i=1, mA
				matrixC(i,j) = matrixC(i,j) + matrixA(i,k)*matrixB(k,j)
			END DO
		END DO
	END DO

END SUBROUTINE multkji

