#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: CHECK_DIR_TREE.SH 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/ls/C/0/0/io_ls/CHECK_DIR_TREE.SH $
# For license details see COPYING in the package base directory
#####################################################################
# Check if directory tree already exist
#####################################################################

# Check syntax

if [ $# != 3 ] ; then
   echo "syntax: $0 <RootDirectoryOfTree> <TreeDepth> <FilesPerLeafDir>"
   exit 127
fi

# Check if directory can be used to write tree

if [ ! -d $1 ] ; then
   echo $1 does not exist or is not a directory
   exit 127
fi

if [ ! -r $1 ] ; then
   echo $1 is not readable
   exit 127
fi

if [ ! -x $1 ] ; then
   echo $1 is not executable
   exit 127
fi

if [ ! -w $1 ] ; then
   echo $1 is not writable
   exit 127
fi

# Check if tree already exist

if [ ! -d $1/d0 -o ! -d $1/d1 ] ; then
    exit 1
fi

# Check if status files are readable

if [ ! -r $1/dir_state -o ! -r $1/reg_files_state ] ; then
    exit 2
fi

# Check if depth of tree is okay

storedDepth=`awk '/^Depth is [0-9]*/ {print substr($3,1,length($3)-1)}' $1/dir_state`
if [ "x$storedDepth" != "x$2" ] ; then
    exit 3
fi

# Check number of files at tree leafs

storedFPD=`awk '/^[0-9]* files per dir/ {print $1}' $1/reg_files_state`
if [ $storedFPD != $3 ] ; then
    exit 4
fi

exit 0


