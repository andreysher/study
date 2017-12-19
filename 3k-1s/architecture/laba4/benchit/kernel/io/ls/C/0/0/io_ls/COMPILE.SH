#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/ls/C/0/0/io_ls/COMPILE.SH $
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

LOCAL_KERNEL_COMPILER="C"
LOCAL_KERNEL_COMPILERFLAGS=""
LOCAL_KERNEL_FILES="read_dirs.c create_dirs.c io_ls_types.c"

LOCAL_MODULE_FILES="kernel_main.c"

LOCAL_BENCHIT_COMPILER="C"
LOCAL_BENCHIT_COMPILERFLAGS=""

LOCAL_LINKERFLAGS=""

cp ${KERNELDIR}/CHECK_OTHER_FILES.SH ${DEST}
cp ${KERNELDIR}/CHECK_DIR_TREE.SH ${DEST}

BENCHIT_COMMENT="Simple C kernel"
export BENCHIT_COMMENT
