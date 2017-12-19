#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/utilities/skeleton/C/MPI/0/simple/COMPILE.SH $
# For license details see COPYING in the package base directory
#####################################################################
# Kernel: simple Variant of the c-Skeleton with MPI
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


LOCAL_KERNEL_COMPILER="MPICC"
LOCAL_KERNEL_COMPILERFLAGS=""
LOCAL_KERNEL_FILES="simple.c kernel_main.c"

LOCAL_MODULE_FILES=""

LOCAL_BENCHIT_COMPILER="MPICC"
LOCAL_BENCHIT_COMPILERFLAGS="${BENCHIT_CPP_MPI}"

LOCAL_LINKERFLAGS=""

BENCHIT_COMMENT="simple C skeleton kernel with MPI"
export BENCHIT_COMMENT