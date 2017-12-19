#!/bin/sh
#####################################################################
# BenchIT - Performance Measurement for Scientific Applications
# Contact: developer@benchit.org
#
# $Id: reference_run.sh 9 2009-09-16 13:24:03Z hoehlig $
# $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/reference_run.sh $
# For license details see COPYING in the package base directory
#####################################################################
# Script which performs a so called BenchIT Reference Run.
# All kernels in kernel/_reference_ will be compiled and
# executed.
# You should edit the c compiler part in your LOCALDEF
# in a proper way before running this script.
# Best run in runlevel 1 ( init 1 ).
#####################################################################

### variables and helpers

# change to BenchIT folder 
cd `dirname $0`

BITPATH=`pwd`				# path to BenchIT root
KERNELPATH=""				# path to kernels, LEAVE EMPTY HERE !
BINPATH=${BITPATH}/bin			# path to kernel binaries
LOG=${BITPATH}/REFRUN.log		# file for logging this run
TMP=${BITPATH}/garbage.tmp		# temporary file needed for resume

# check if path to binaries exists
if [ ! -d ${BINPATH} ]; then
	mkdir -v ${BINPATH}
fi

STATS=0
RUNLEVEL=`who -r | tr -s " " | cut -f3 -d " "`	# current runlevel
TIMECMD=`which time`				# check if time command is available

# publishing signal handler
trap 'handle_SIGINT' 2


### function declaration

# what to do if user interrupts?:
handle_SIGINT ()
{
	printf "\n\033[1;31m...caught SIGINT: exiting!\033[0m\n"
	rm -f ${TMP}
	exit 1
}


# this function will be run if no arguments are given
# TODO: maybe add some error handling
normal_run ()
{
	prepare		# clean up, etc.
	compile		# compile all reference kernels
	run				# run all compiled kernels
	epilog		# write final status data
	
	exit 0
}


# This function will be executed by the -r argument.
# It tries to find out where was interrupted last time and resumes if possible.
# recompiles all if there were compile errors before
resume ()
{
	runlevel_check

	printf "\033[1;34m try to resume last reference run...\n\033[0m"
	
	# was there a reference run before?
	if [ ! -e ${LOG} ]; then
		printf "\033[1;34m No logfile found, cannot resume!\nTry a normal run!\033[0m"
		exit 1
	fi

	egrep "\<#####\>" ${LOG} | tr -d "#" | tr -d " " > ${TMP}
	COUNT=`wc -l ${TMP} | cut -f1 -d " "`	# number of started kernels
	if [ ${COUNT} -lt 1 ]; then
		printf "\033[1;34m ...seems like there were no kernels run before\n...recompiling all in 3 sec.\n\033[0m"
		sleep 3
		normal_run
		exit 0
	fi

	LAST=`tail -n 1 ${TMP}`
	printf "\033[1;34mIt seems\033[0m ${LAST} \033[1;34mwas the interrupted kernel\n\033[0m"

	# deleting correct proceeded kernels
	printf "\033[1;34m...deleting correct proceeded kernels\n\033[0m"
	sleep 2
	
	for i in `cat ${TMP}`; do
		if [ ${COUNT} -lt 2 ]; then
			break
		fi

		printf "\033[1;34m...removing\033[0m ${BINPATH}/$i \n"
		rm -f "${BINPATH}/$i"
		COUNT=$((${COUNT}-1))
	done

	# clean up old log data
	sed "/#####/d" ${LOG} > ${TMP}
	cat ${TMP} > ${LOG}

	echo "...resuming" >> ${LOG}
	run
	epilog

	exit 0
}


# The printout when -h or a wrong parameter is given.
print_help ()
{
	printf "Script which performs a so called BenchIT Reference Run.\n\n"
	
	printf "\tusage: $0 [[-a|-p PATH] [-r] [-s] [-h]]\n\n"

	printf "no option:\tstart a fresh reference run\n"

	printf " -a:\t\tproceed ALL kernels, not just the reference\n"
	printf "\t\t\tWARNING this will need much time!)\n"

	printf " -p KERNELPATH:\tproceed all kernels in KERNELPATH\n"

	printf " -r:\t\ttry to resume a previouse interrupted session,\n"
	printf "\t\t\t${LOG} is needed!\n"

	printf " -s:\t\twrite advanced runtime statistic to\n"
	printf "\t\t\t${LOG}\n"

	printf " -h:\t\tprint this help\n\n"
	exit 0
}


# checks the current runlevel and gives possibility to cancel
runlevel_check ()
{
	if [ "${RUNLEVEL}" != "1" ] && [ "${RUNLEVEL}" != "S" ] && [ "${RUNLEVEL}" != "s" ]; then
		printf "\033[1;31mIt is recommend to run all kernels in runlevel 1, S or s!\nYour current runlevel is:\033[0m ${RUNLEVEL} \n"
#		echo "Proceed anyway? (y/n): "
#		read ANSWER
#		while [ "${ANSWER}" != "y" ] && [ "${ANSWER}" != "n" ]; do
#			printf "\033[1;31mI didn't understand you.\n\033[0m"
#			printf "Proceed anyway? (y/n): "
#			read ANSWER
#		done
#
#		if [ "${ANSWER}" = "n" ]; then
#			echo "... exiting"
#			exit 0
#		fi
		sleep 2
		
	fi

	return 0
}



# STAGE 0: preparing
# normally not run stand alone
prepare ()
{
	rm -f ${LOG}
	printf "\033[1;32mSTAGE 0: preparing...\n\033[0m"

	runlevel_check

	rm -rf ${BINPATH}/*		# necessary cleanup

	echo "### BenchIT Reference Run ###" > ${LOG}
	echo "Date: `date`" >> ${LOG}
	echo "-----------------------------" >> ${LOG}

	return 0
}


# STAGE 1: compling
# normally not run stand alone
compile ()
{
	ERRORS=0
	printf "\033[1;32mSTAGE 1: compiling...\n\033[0m"
	echo "Compiler warnings/errors per kernel" >> ${LOG}

	for i in `find ${KERNELPATH} -name PARAMETERS`; do
		echo `dirname $i` >> ${LOG}
		rm -f ${TMP}
		./COMPILE.SH `dirname $i` 2> ${TMP}
		if [ ! $? -eq 0 ] || [ -s ${TMP} ]; then
			if [ "`egrep "\<[Ee]rror\>" ${TMP}`" ]; then
				cat ${TMP} >> ${LOG}
				ERRORS=$((${ERRORS}+1))
			fi
			cat ${TMP} >> ${LOG}
		fi
	done
	
	# handle errors while compiling
	if [ ${ERRORS} -gt 0 ]; then
		printf "\033[1;31m ${ERRORS} Error(s) appeared during compilation of `dirname $i`!\nSee\033[0m ${LOG} \033[1;31mfor further information\033[0m\n\n"
		sleep 2

		while :; do
			printf "Do you want to proceed with what was compiled? (y/n): "
			read ANSWER
			if [ "${ANSWER}" = "y" ]; then
				return 0
			elif [ "${ANSWER}" = "n" ]; then
				printf "\nexiting...\n"
				exit 1
			fi

			printf "I don't understand \"${ANSWER}\"!\n\n"
		done
	fi

	return 0
}


# STAGE 2: running
# normally not run stand alone
run ()
{
	printf "\033[1;32mSTAGE 2: running...\n\033[0m"
	echo "Status information to kernel runs:" >> ${LOG}
	
	#   create run command
	if [ -z ${TIMECMD} ] || [ ${STATS} -eq 0 ]; then
		RUNCMD=./RUN.SH
		if [ -z ${TIMECMD} ] && [ ${STATS} -eq 1 ]; then
			printf "\033[1;31mYou need the time command on your system for advanced runtime statistic!\n\033[0;"
			sleep 2
		fi
		
	else
		RUNCMD="${TIMECMD} -a -o ${LOG} -f \"Command: %C\nElapsed wall clock time: %E s\nCPU time in kernel mode:\t%S s\nCPU time in user mode:\t%U s\nCPU time in percent: %P\nData usage: %D Kb\nAvg. total memory: %K Kb\nMax used memory: %M Kb\nStack size (Kb): %p\nPage faults (real): %F\nPage faults (without I/O): %R\nFS inputs: %I\nFS outputs: %O\nSwaped: %W times\nContext switches: %c\nContext switches (voluntary): %w\nSignals received: %k\nSocket msg received: %r\nSocket msg send: %s\nExited with: %x\" ./RUN.SH"

	fi

	#   main run loop
	NUM=`ls ${BINPATH} | egrep -v ".+\.[sS][hH]\>" | wc -l`
	COUNT=1			# the current executing kernel counter
		
	for i in `ls ${BINPATH} | egrep -v ".+\.[sS][hH]\>"`; do
		echo "##### $i #####" >> ${LOG}
		printf "\033[1;32m##### $i (${COUNT}/${NUM}) #####\n\033[0m"
		COUNT=$((${COUNT}+1))
		eval "${RUNCMD} ${BINPATH}/$i"
		if [ ! $? -eq 0 ]; then
			printf "\033[1;31mErrors appeared during execution of $i!\nSee\033[0m ${LOG} \033[1;31mfor further information\n\n"
			printf "...exiting\n"
			exit 1
		fi
	done

	return 0
}



# STAGE 3: epilog
# normally not run stand alone
epilog ()
{
	printf "\033[1;32mSTAGE 3: epilog...\n\033[0m"
	echo "Reference Run ended `date`" >> ${LOG}

	WARNINGS=`egrep "\<[Ww]arning\>" ${LOG} | wc -l | cut -f1 -d " "`
	ERRORS=`egrep "\<[Ee]rror\>" ${LOG} | wc -l | cut -f1 -d " "`
	if [ ${ERRORS} -gt 1 ]; then
	    echo "${ERRORS} error(s) during reference run!" >> ${LOG}
		printf "\033[1;31m${ERRORS} error(s) during reference run!\nPlease check ${LOG} for further information.\n\033[0m"
	fi
	
	if [ ${WARNINGS} -gt 1 ]; then
	    echo "${WARNINGS} warning(s) during reference run!" >> ${LOG}
		printf "\033[1;31m${WARNINGS} warning(s) during reference run!\nPlease check ${LOG} for further information.\n\033[0m"
	fi

	# short cleanup
	rm -f ${TMP}

	return 0
}


##############################################################################
### main part

# proceeding arguments:
while getopts "ap:rsh" OPT ; do
        case ${OPT} in
          a) if [ -z "${KERNELPATH}" ]; then 
		KERNELPATH=${BITPATH}/kernel
  	 else
		printf "\033[1;31mERROR: path to kernels already set\033[0m\n"
		printf "\033[1;31mThis appears when using -p and -a option together\033[0m\n"
		exit 1
	fi ;;
          p) if [ -z "${KERNELPATH}" ]; then 
		KERNELPATH=${OPTARG}
	else
		printf "\033[1;31mERROR: path to kernels already set\033[0m\n"
		printf "\033[1;31mThis appears when using -p and -a option together\033[0m\n"
		exit 1
	fi ;;
          r) resume ;;
          s) STATS=1 ;;
	  h|help) print_help ;;
          *) print_help ;;
        esac
done

# just a normal reference run
if [ -z "${KERNELPATH}" ]; then 
	KERNELPATH=${BITPATH}/kernel/_reference_
	if [ ! -d ${KERNELPATH} ]; then
		printf "\033[1;31mERROR: path to kernels ( ${KERNELPATH} ) does not exist!\033[0m\n"
		exit 1
	fi
fi

# no arguments given?:
# start normal run
normal_run


exit 0
