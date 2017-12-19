#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: GUI.sh 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/GUI.sh $
# For license details see COPYING in the package base directory
#####################################################################

# change to the BenchIT Directory
cd `dirname ${0}` || exit 1

cd gui/bin
./GUI.sh
