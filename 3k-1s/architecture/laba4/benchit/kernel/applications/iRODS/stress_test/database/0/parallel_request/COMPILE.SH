#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: benchit@zih.tu-dresden.de
#
# Last change by: $Author: william $
# $Revision: 1.8 $
# $Date: 2006/07/01 15:42:45 $
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


LOCAL_KERNEL_COMPILER=""
LOCAL_KERNEL_COMPILERFLAGS=""
LOCAL_KERNEL_FILES=""

LOCAL_MODULE_FILES="kernel_main.c"

LOCAL_BENCHIT_COMPILER="C"
LOCAL_BENCHIT_COMPILERFLAGS=""

LOCAL_LINKERFLAGS=""

BENCHIT_COMMENT="Simple C kernel"
export BENCHIT_COMMENT