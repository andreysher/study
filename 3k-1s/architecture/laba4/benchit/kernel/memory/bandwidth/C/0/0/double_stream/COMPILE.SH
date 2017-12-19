#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/bandwidth/C/0/0/double_stream/COMPILE.SH $
# For license details see COPYING in the package base directory
#####################################################################
# Kernel: measure Bandwidth inspired by STREAM benchmark (C-version)
#
#         according to the rules, reffer this Benchmark as:
#         "BenchIT kernel based on a variant of the STREAM benchmark code"
#         when publishing results.
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
LOCAL_KERNEL_FILES="work.c"

LOCAL_MODULE_FILES="kernel_main.c"

LOCAL_BENCHIT_COMPILER="C"
LOCAL_BENCHIT_COMPILERFLAGS=""

LOCAL_LINKERFLAGS=""