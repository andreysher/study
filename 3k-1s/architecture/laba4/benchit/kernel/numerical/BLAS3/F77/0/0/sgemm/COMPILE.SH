#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/BLAS3/F77/0/0/sgemm/COMPILE.SH $
# For license details see COPYING in the package base directory
#####################################################################
# Kernel: Measurment of dgemv performance
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


LOCAL_KERNEL_COMPILER="C"
LOCAL_KERNEL_COMPILERFLAGS="-c -DADD_"
LOCAL_KERNEL_FILES="cblas_sgemm.c cblas_xerbla.c cblas_globals.c"

LOCAL_MODULE_FILES="kernel_main.c"

LOCAL_BENCHIT_COMPILER="C"
LOCAL_BENCHIT_COMPILERFLAGS=""

LOCAL_LINKERFLAGS=""

BENCHIT_COMMENT="sgemm"
export BENCHIT_COMMENT

# FOLLOWING: Unprocessed commands
# TODO: Check if you need them

BENCHIT_COMMENT="sgemm"
export BENCHIT_COMMENT
TOOLSDIR="${BENCHITROOT}/tools"
if [ -f "${TOOLSDIR}/BLAS/sgemm.f" ] && [ -f "${TOOLSDIR}/BLAS/lsame.f" ] && [ -f "${TOOLSDIR}/BLAS/xerbla.f" ] && [ -f "${TOOLSDIR}/CBLAS/src/cblas_globals.c" ] && [ -f "${TOOLSDIR}/CBLAS/src/cblas_xerbla.c" ] && [ -f "${TOOLSDIR}/CBLAS/src/cblas_sgemm.c" ] && [ -f "${TOOLSDIR}/CBLAS/src/cblas.h" ] && [ -f "${TOOLSDIR}/CBLAS/src/cblas_f77.h" ]; then
	cp ${TOOLSDIR}/BLAS/sgemm.f              ${KERNELDIR}
	cp ${TOOLSDIR}/BLAS/lsame.f              ${KERNELDIR}
	cp ${TOOLSDIR}/BLAS/xerbla.f             ${KERNELDIR}
	cp ${TOOLSDIR}/CBLAS/src/cblas_globals.c ${KERNELDIR}
	cp ${TOOLSDIR}/CBLAS/src/cblas_xerbla.c  ${KERNELDIR}
	cp ${TOOLSDIR}/CBLAS/src/cblas_sgemm.c   ${KERNELDIR}
	cp ${TOOLSDIR}/CBLAS/src/cblas.h         ${KERNELDIR}
	cp ${TOOLSDIR}/CBLAS/src/cblas_f77.h     ${KERNELDIR}
else
	echo "At least one of the files:"
	echo "  ${TOOLSDIR}/BLAS/sgemm.f"
	echo "  ${TOOLSDIR}/BLAS/lsame.f"
	echo "  ${TOOLSDIR}/BLAS/xerbla.f"
	echo "  ${TOOLSDIR}/CBLAS/src/cblas_globals.c"
	echo "  ${TOOLSDIR}/CBLAS/src/cblas_xerbla.c"
	echo "  ${TOOLSDIR}/CBLAS/src/cblas_sgemm.c"
	echo "  ${TOOLSDIR}/CBLAS/src/cblas.h"
	echo "  ${TOOLSDIR}/CBLAS/src/cblas_f77.h"
	echo "is not available, check Netlib BLAS Library in ${TOOLSDIR}"
	exit 1
fi
BENCHIT_COMPILER="${BENCHIT_CC}"
BENCHIT_COMPILERFLAGS=""
BENCHITC_COMPILER="${BENCHIT_CC}"
C_COMPILE="${BENCHIT_COMPILER} ${BENCHIT_COMPILERFLAGS} -c"
BENCHITF_COMPILERFLAGS=""
F_COMPILE="${BENCHIT_F77} ${BENCHITF_COMPILERFLAGS} -c"
L_FLAGS=""
F_LINK="${BENCHIT_F77} ${BENCHITF_COMPILERFLAGS} -o ${BENCHIT_KERNELBINARY} *.o ${L_FLAGS}"
export BENCHIT_COMPILER BENCHIT_COMPILERFLAGS L_FLAGS
${F_COMPILE} sgemm.f
${F_COMPILE} lsame.f
${F_COMPILE} xerbla.f
${C_COMPILE} -DADD_ cblas_globals.c
${C_COMPILE} -DADD_ cblas_xerbla.c
${C_COMPILE} -DADD_ cblas_sgemm.c
${C_COMPILE} kernel_main.c
${BENCHITC_COMPILER} -c ${BENCHITROOT}/benchit.c
${F_LINK}
if [ ! $? -eq 0 ]; then
	echo "If you have a undefined reference to a BLAS function, check COMPILE.sh and replace the default -DADD_ flag with one of the flags -DUPCASE or -DNOCHANGE"
   exit 1
fi
rm -f ${KERNELDIR}/*.f ${KERNELDIR}/cblas_*.c ${KERNELDIR}/cblas.h ${KERNELDIR}/cblas_f77.h
