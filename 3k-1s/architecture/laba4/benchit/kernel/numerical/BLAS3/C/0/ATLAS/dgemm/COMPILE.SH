#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/BLAS3/C/0/ATLAS/dgemm/COMPILE.SH $
# For license details see COPYING in the package base directory
#####################################################################
# Kernel: Measurment of dgemm performance
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


if [ -z "${BENCHIT_CPP_ATLAS}" ] || [ -z "${BENCHIT_LIB_ATLAS}" ]; then
	echo "No include or library path for ATLAS specified - check \$BENCHIT_CPP_ATLAS and \$BENCHIT_LIB_ATLAS"
	exit 1
fi

LOCAL_KERNEL_COMPILER="NONE"
LOCAL_KERNEL_COMPILERFLAGS=""
LOCAL_KERNEL_FILES=""

LOCAL_MODULE_FILES="kernel_main.c"

LOCAL_BENCHIT_COMPILER="C"
LOCAL_BENCHIT_COMPILERFLAGS="${BENCHIT_CPP_ATLAS} -DUSE_ATLAS"

LOCAL_LINKERFLAGS="${BENCHIT_LIB_ATLAS}"
