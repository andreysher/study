#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/BLAS2/F77/0/0/sgemv/COMPILE.SH $
# For license details see COPYING in the package base directory
#####################################################################
# Kernel: Measurment of sgemv performance
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

LOCAL_KERNEL_COMPILER="F77"
LOCAL_KERNEL_COMPILERFLAGS=""
LOCAL_KERNEL_FILES="sgemv.f lsame.f xerbla.f"

LOCAL_MODULE_FILES="kernel_main.c cblas_globals.c cblas_xerbla.c cblas_sgemv.c"

LOCAL_BENCHIT_COMPILER="C"
# If you have a undefined reference to a BLAS function, replace the default -DADD_ flag with one of the flags -DUPCASE or -DNOCHANGE
LOCAL_BENCHIT_COMPILERFLAGS="-DADD_"

LOCAL_LINKERFLAGS=""

# netlib blas library
TOOLSDIR="${BENCHITROOT}/tools/"
BLASSRC="${TOOLSDIR}CBLAS/src/"
LOCAL_FILES_TO_COPY="${TOOLSDIR}BLAS/sgemv.f ${TOOLSDIR}BLAS/lsame.f ${TOOLSDIR}BLAS/xerbla.f ${BLASSRC}cblas_globals.c ${BLASSRC}cblas_xerbla.c"
LOCAL_FILES_TO_COPY="${LOCAL_FILES_TO_COPY} ${BLASSRC}cblas_sgemv.c ${BLASSRC}cblas.h ${BLASSRC}cblas_f77.h"
LOCAL_FILES_TO_DELETE="*.f cblas_*.c cblas.h cblas_f77.h"


BENCHIT_COMMENT="sgemv"
export BENCHIT_COMMENT
