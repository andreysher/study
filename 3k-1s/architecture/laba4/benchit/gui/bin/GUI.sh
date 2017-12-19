#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: GUI.sh 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/gui/bin/GUI.sh $
# For license details see COPYING in the package base directory
#####################################################################
# Shell script, starting the gui
#####################################################################

OSNAME=`uname`
cd `dirname ${0}`
cd ..
GUIDIR=`pwd`
cd ..
BENCHITDIR=`pwd`
cd "${GUIDIR}/cfg"
CFGDIR=`pwd`


cd "$GUIDIR/bin"
### get hostname if not given
if test -z "$NODENAME" ; then
   NODENAME="`hostname`" || NODENAME="`uname -n`"
     NODENAME=${NODENAME%%.*}
   
fi

 set +e
# if we are new on the machine (LOCALDEFS don't exist) run script FIRSTTIME
if test -f "${BENCHITDIR}/LOCALDEFS/${NODENAME}" ; then
   echo "transfer all money to" > /dev/null
else
   echo "No LOCALDEFS for host ${NODENAME} found. Starting FIRSTTIME..."
  export BENCHIT_INTERACTIVE="0"
   ${BENCHITDIR}/tools/FIRSTTIME
fi

### execute java GUI

ARGS="-host=$NODENAME $*"
if [ "${OSNAME}" = "Darwin" ]; then
	java -Xdock:name="BenchIT" -Xdock:icon=../img/splash.jpg -Xmx256m -Xms256m -classpath BenchIT.jar -Dusessl=true -Djavax.net.ssl.trustStore=../cfg/client.trusts -Djavax.net.ssl.trustStorePassword=BenchIT BIGMain $ARGS
else
	java -Xmx1024m -Xms1024m -classpath BenchIT.jar -Dusessl=true -Djavax.net.ssl.trustStore=../cfg/client.trusts -Djavax.net.ssl.trustStorePassword=BenchIT BIGMain $ARGS
fi

RETURNCODE=$?
echo "Returncode was $RETURNCODE"


