#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/bandwidth/C/OpenMP/0/double_stream_opt/COMPILE.SH $
# For license details see COPYING in the package base directory
#####################################################################
# Kernel: measure Bandwidth inspired by STREAM benchmark (C OMP-version)
#
#         according to the rules, reffer this Benchmark as:
#         "BenchIT kernel based on a variant of the STREAM benchmark code"
#         when publishing results
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

BENCHIT_PARAMETER_FLAGS=""

if [ "${BENCHIT_KERNEL_ENABLE_NONTEMPORAL_STORES}" -ne 0 ]; then
  BENCHIT_PARAMETER_FLAGS="${BENCHIT_PARAMETER_FLAGS} -DBENCHIT_KERNEL_ENABLE_NONTEMPORAL_STORES"
fi

if [ "${BENCHIT_KERNEL_ENABLE_ALIGNED_ACCESS}" -ne 0 ]; then
  BENCHIT_PARAMETER_FLAGS="${BENCHIT_PARAMETER_FLAGS} -DBENCHIT_KERNEL_ENABLE_ALIGNED_ACCESS"
fi

if [ "${BENCHIT_KERNEL_COMPILE_FOR_PIN_THREADS_TO_CORES}" -ne 0 ]; then
  BENCHIT_PARAMETER_FLAGS="${BENCHIT_PARAMETER_FLAGS} -DBENCHIT_KERNEL_COMPILE_FOR_PIN_THREADS_TO_CORES"
fi

echo "\$BENCHIT_PARAMETER_FLAGS=${BENCHIT_PARAMETER_FLAGS}"


LOCAL_KERNEL_COMPILER="C"
LOCAL_KERNEL_COMPILERFLAGS="${BENCHIT_CC_C_FLAGS_OMP} ${BENCHIT_PARAMETER_FLAGS}"
LOCAL_KERNEL_FILES="work.c"

LOCAL_MODULE_FILES="kernel_main.c"

LOCAL_BENCHIT_COMPILER="C"
LOCAL_BENCHIT_COMPILERFLAGS="${BENCHIT_CC_C_FLAGS_OMP} ${BENCHIT_PARAMETER_FLAGS}"

LOCAL_LINKERFLAGS="${BENCHIT_CC_C_FLAGS_OMP}"
