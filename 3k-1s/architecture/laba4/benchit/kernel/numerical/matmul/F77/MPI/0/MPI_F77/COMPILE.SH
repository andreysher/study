#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/F77/MPI/0/MPI_F77/COMPILE.SH $
# For license details see COPYING in the package base directory
#####################################################################

# If it's not set, the user
# executed this script directly so we need to call top script
if [ -z "${BENCHIT_COMPILE_SH_IS_CALLED}" ]; then
   # Go to COMPILE.SH directory
   cd `dirname ${0}` || exit 1

   # compile using global script
   ../../../../../../../COMPILE.SH `pwd`
   exit 0
fi


LOCAL_KERNEL_COMPILER="MPIF77"
LOCAL_KERNEL_COMPILERFLAGS=""
LOCAL_KERNEL_FILES="mpi_matrix_functions.f mpi_functions.f matrix_functions.f matmul_f_core.f"

LOCAL_MODULE_FILES="matmul_sub.c"

LOCAL_BENCHIT_COMPILER="MPICC"
LOCAL_BENCHIT_COMPILERFLAGS="${BENCHIT_CPP_MPI}"

LOCAL_LINKERFLAGS=""

BENCHIT_COMMENT="Matrix Multiplication in F77 with MPI (F77)"
export BENCHIT_COMMENT