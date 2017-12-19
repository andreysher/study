#!/bin/sh

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
LOCAL_KERNEL_COMPILERFLAGS="${BENCHIT_F77_C_FLAGS_OMP}"
LOCAL_KERNEL_FILES="matmul_f_core.f"

LOCAL_MODULE_FILES="matmul_sub.c"

LOCAL_BENCHIT_COMPILER="C"
LOCAL_BENCHIT_COMPILERFLAGS="${BENCHIT_CC_C_FLAGS_OMP}"

LOCAL_LINKERFLAGS="${BENCHIT_F77_C_FLAGS_OMP}"