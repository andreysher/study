#!/bin/sh
######################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: COMPILE.SH 1 2011-01-29 fschmitt $
# $URL: svn+ssh://benchit@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/FFT_2D/C/CUDA/CUFFT/complex_double_npo2/COMPILE.SH $
# For license details see COPYING in the package base directory
#####################################################################
# Kernel: 2D Fast Fourier Transform, Non-Powers of 2,
# double precision, complex data, CUFFT
# (C)
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

LOCAL_KERNEL_COMPILER="NONE"
LOCAL_KERNEL_COMPILERFLAGS=""
LOCAL_KERNEL_FILES=""

LOCAL_MODULE_FILES="kernel_main.c"

LOCAL_BENCHIT_COMPILER="C"
LOCAL_BENCHIT_COMPILERFLAGS="-I${CUDA_INC}"

LOCAL_LINKERFLAGS="-L${CUDA_LIB} -lcufft -lcudart"
