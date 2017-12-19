#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/BLAS1/F77/0/0/daxpy/COMPILE.SH $
# For license details see COPYING in the package base directory
#####################################################################
# Kernel: Measurment of daxpy performance
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
LOCAL_KERNEL_FILES="daxpy.f"

LOCAL_MODULE_FILES="kernel_main.c cblas_daxpy.c"

LOCAL_BENCHIT_COMPILER="C"
# If you have a undefined reference to a BLAS function, replace the default -DADD_ flag with one of the flags -DUPCASE or -DNOCHANGE
LOCAL_BENCHIT_COMPILERFLAGS="-DADD_"

LOCAL_LINKERFLAGS=""

# netlib blas library
TOOLSDIR="${BENCHITROOT}/tools"
LOCAL_FILES_TO_COPY="${TOOLSDIR}/BLAS/daxpy.f ${TOOLSDIR}/CBLAS/src/cblas_daxpy.c ${TOOLSDIR}/CBLAS/src/cblas.h ${TOOLSDIR}/CBLAS/src/cblas_f77.h"
LOCAL_FILES_TO_DELETE="daxpy.f cblas_daxpy.c cblas.h cblas_f77.h"

BENCHIT_COMMENT="daxpy"
export BENCHIT_COMMENT
