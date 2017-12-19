#!/bin/sh
###############################################################################
#
#  B e n c h I T - Performance Measurement for Scientific Applications
#
#  Shell script, installing the gui
#
#  Author: SWTP Nagel 1
#  Last change by: $Author: domke $
#  $Revision: 1.2 $
#  $Date: 2008/05/29 12:06:07 $
#
###############################################################################
echo "###### installing ######"

cd ../src
jar uf ../bin/BenchIT.jar *.class */*.class org/syntax/jedit/tokenmarker/*.class
rm *.class
rm */*.class
rm org/syntax/jedit/tokenmarker/*.class
cd ../bin

echo "###### done       ######"
###############################################################################
#  Log-History
#
###############################################################################
