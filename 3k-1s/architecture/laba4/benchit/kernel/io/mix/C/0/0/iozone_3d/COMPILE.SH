#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/mix/C/0/0/iozone_3d/COMPILE.SH $
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

printf "\n\ncompile iozone:\n\n"
cd ${KERNELDIR}/src/iozone
make clean
make ${IOZONE_COMPILE_ARCH}
cp ./iozone ${BENCHITROOT}/bin/

LOCAL_KERNEL_COMPILER=""
LOCAL_KERNEL_COMPILERFLAGS=""
LOCAL_KERNEL_FILES=""

LOCAL_MODULE_FILES="kernel_main.c"

LOCAL_BENCHIT_COMPILER="C"
LOCAL_BENCHIT_COMPILERFLAGS=""

LOCAL_LINKERFLAGS=""

BENCHIT_COMMENT="iozone wrapper kernel"
export BENCHIT_COMMENT
