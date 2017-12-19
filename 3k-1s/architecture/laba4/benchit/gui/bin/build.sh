#!/bin/sh
###############################################################################
#
#  B e n c h I T - Performance Measurement for Scientific Applications
#
#  Shell script, compiling the gui
#
#  Author: SWTP Nagel 1
#  Last change by: $Author: domke $
#  $Revision: 1.2 $
#  $Date: 2008/05/29 12:06:07 $
#
###############################################################################
echo "###### compiling  ######"

OLDDIR="${PWD}"
cd `dirname ${0}`
javac ${1} -classpath BenchIT.jar ../src/*.java ../src/system/*.java ../src/gui/*.java ../src/admin/*.java ../src/plot/*.java ../src/conn/*.java ../src/org/syntax/jedit/tokenmarker/*.java

./install.sh

cd "${PWD}"
echo "###### done       ######"
###############################################################################
#  Log-History
#
###############################################################################
