#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: CHECK_OTHER_FILES.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/ls/C/0/0/io_ls/CHECK_OTHER_FILES.SH $
# For license details see COPYING in the package base directory
#####################################################################
# Check if files in other than leaf directories already exist
#####################################################################

# Check syntax

if [ $# != 3 ] ; then
   echo "syntax: $0 <RootDirectoryOfTree> <TreeDepth> <FilesPerLeafDir>"
   exit 127
fi

# If status file does not exit
if [ ! -r $1/reg_files_interval_state ] ; then
    exit 2
fi

# Check depth interval
depthInt=`grep Depth $1/reg_files_interval_state|sed -e 's/.*Depth is \([0-9]*-[0-9]*\)\..*/\1/'`
if [ "1-`expr $2 - 1`" != "$depthInt" ]; then
    exit 3
fi

exit 0
