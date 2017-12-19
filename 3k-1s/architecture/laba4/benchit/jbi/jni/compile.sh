#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: compile.sh 1 2009-09-11 12:26:19Z william $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/jbi/jni/compile.sh $
# For license details see COPYING in the package base directory
#####################################################################
# this script compiles an additional C-Library for better precision 
# when testing Java-kernels
#
# Usage
# ./compile.sh [SDK-LOCATION]
#
# ! Java-kernels should not rely on C-libraries being available, it 
# is supposed to use the Java-wrapper-methods
#   in class JBI as they provide fallbackimplementations in Java if 
 #the libraries are missing
#
# ! if you add new native methods to ${BENCHITROOT}/jbi/JBI.java you 
# have to recreate the JBI.h by:
# cd ${BENCHITROOT}/jbi/
# javac JBI.java
# javah JBI
# cp JBI.h jni
#####################################################################


# Compiles C-Library
compile(){
        OLDDIR="`pwd`"
        cd ${2}
        printf "         compiling JNI-Library\n"
        DIRECTORIES="`find \"${1}/include\" -type d`"
        DIRECTORIES="`echo -n ${DIRECTORIES}`"
        INCLUDE=""
        IFS=' '
        for D in ${DIRECTORIES}
        do
                INCLUDE="${INCLUDE} -I${D}"
        done

        ${JNI_CC} -shared -o libjbi.so -fPIC ${INCLUDE} JBI.c

        cd ${OLDDIR}
}


# try to find Java-SDK if not specified or invalid
find_java(){

        #first attempt is to read BENCHIT_JAVA_HOME from Localdefs
	if [ -n ${BENCHIT_JAVA_HOME} ] && [ -d "${BENCHIT_JAVA_HOME}/include" ]
        then
        	compile "${BENCHIT_JAVA_HOME}" "${JNI_DIR}"
                exit 0
        fi

        #next we check if JAVA_HOME environment variable is set and actually points to a SDK
        if [ -n ${JAVA_HOME} ] && [ -d "${JAVA_HOME}/include" ]
        then
                compile "${JAVA_HOME}" "${JNI_DIR}"
                exit 0
        fi



        #last chance is to specify SDK-location manually
        #printf "         Automatic detection failed. Would you like to specify SDK-Location on your own? (y/n) "
        #read e1
        #if [ "${e1}" = "y" ]; then
        #        printf "         SDK-Location: "
        #        read JAVA_DIR
        #        if [ -n ${JAVA_DIR} ] && [ -d "${JAVA_DIR}/include" ]
        #        then
        #                printf "          - OK\n"
        #                compile "${JAVA_DIR}" "${JNI_DIR}"
        #                exit 0
        #        else
        #                printf "          - Failed. ${JAVA_DIR} does not seem to contain a Java-SDK \n"
        #                echo "         No Java-SDK could be found. Can't compile C-Libraries."
        #                # Java-fallbacks will be used
        #                exit -1
        #        fi
        #
        #else
                echo "         Warning: could not compile optional JNI-Library for the Java interface."
                echo "                  make sure the JAVA_HOME variable is set correctly,"
                echo "                  and run BENCHITROOT/jbi/jni/compile.sh manually if you need them"
                exit -1
                # Java-fallbacks will be used
        #fi
}

#Main
JNI_DIR="`dirname \"${0}\"`"
BENCHITROOT="${JNI_DIR}/../.."

HOSTNAME="`hostname`"
if  [ -f "${JNI_DIR}/../../LOCALDEFS/${HOSTNAME}" ]; then
	set -a
        . "${JNI_DIR}/../../LOCALDEFS/${HOSTNAME}"
	set +a
fi

#use BENCHIT_CC by default
if [ ${BENCHIT_CC} ] && [ -n ${BENCHIT_CC} ]; then
  JNI_CC=${BENCHIT_CC}
else
  # choose C-Compiler (when called standalone without LOCALDEFS)
  if "${BENCHITROOT}/tools/features" have icc ;then
        JNI_CC="icc"
  elif "${BENCHITROOT}/tools/features" have pathcc ;then
        JNI_CC="pathcc"
  elif "${BENCHITROOT}/tools/features" have pgcc ;then
        JNI_CC="pgcc"
  elif "${BENCHITROOT}/tools/features" have cc ;then
        JNI_CC="cc"
  elif "${BENCHITROOT}/tools/features" have gcc ;then
        JNI_CC="gcc"
  else
   echo "         Warning: could not compile optional JNI-Library for the Java interface."
   echo "                  set the BENCHIT_CC variable to your favorite compiler, export it,"
   echo "                  and run BENCHITROOT/jbi/jni/compile.sh manually if you need them"
  fi
fi

if [ ${1} ]
then

        # if a parameter is specified it is supposed to be a path to a Java-SDK
        if [ -d "${1}/include" ]; then
                compile "${1}" "${JNI_DIR}"
        else    # well supposed to be
                find_java
        fi


else
        find_java
fi
