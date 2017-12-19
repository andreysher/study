#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
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

LOCAL_KERNEL_COMPILER="CUDA"
LOCAL_KERNEL_COMPILERFLAGS=""
LOCAL_KERNEL_FILES=""

LOCAL_MODULE_FILES="kernel_main.cu"

LOCAL_BENCHIT_COMPILER="CUDA"
LOCAL_BENCHIT_COMPILERFLAGS=""
if [ `hostname` = "wildcard" ] || [ `hostname` = "joker" ] || [ `hostname` = "playground" ]; then
  LOCAL_BENCHIT_COMPILERFLAGS="${LOCAL_BENCHIT_COMPILERFLAGS} -arch=sm_13"
fi

LOCAL_LINKERFLAGS=""

BENCHIT_COMMENT="CUDA skeleton kernel"
export BENCHIT_COMMENT
