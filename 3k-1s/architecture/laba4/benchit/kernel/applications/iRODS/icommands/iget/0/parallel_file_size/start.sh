#! /bin/sh

# $1 = Increment of filesize
# $2 = Unit
# $3 = Number of runs

FILE_SIZE=$1
FILE=$BENCHIT_RESULT_NAME
SPECIAL="benchit_$1"
SUBDIR="benchit_subdir_size"
PATH_TEMP=$BENCHIT_KERNEL_PATH_TEMP
PATH_SCRIPT=$KERNELDIR
IRODS_RESOURCE=""
IRODS_PROT=""
IRODS_THREADS=""
DD_UNIT=""
DD_BLOCK=1
BOOL=1

if [ "$BENCHIT_IRODS_RESC" != "" ]
then
	IRODS_RESOURCE="-R $BENCHIT_IRODS_RESC"
fi

if [ "$BENCHIT_IRODS_PROT" -ne 0 ]
then
	IRODS_PROT="-Q"
fi
if [ "$BENCHIT_IRODS_THREADS" -ne -1 ]
then
	IRODS_THREADS="-N $BENCHIT_IRODS_THREADS"
fi
if [ "$BENCHIT_KERNEL_FILE_SIZE_INC_FUNC" -eq 0 ]
then
	DD_UNIT="$BENCHIT_KERNEL_SMALL_UNIT"
else
	if [ "$2" = "B"  ]
  then
		DD_UNIT=""
	else
	  DD_UNIT="$2"
	fi
	SPECIAL="benchit_$1_$2"
fi
if [ "$DD_UNIT" = "G" ]
then
  FILE_SIZE=1
	DD_BLOCK="$1"
fi

PATH_TEMP="$PATH_TEMP/$SUBDIR/"
HELP="$PATH_TEMP*"

# Creates new files if they not already exist and transfers them to iRODS
if [ ! -d $PATH_TEMP ]
then
	mkdir $PATH_TEMP
  imkdir $SUBDIR	
  
	dd if=/dev/urandom of="$PATH_TEMP/$SPECIAL" bs="$FILE_SIZE$DD_UNIT" count="$DD_BLOCK" 2>/dev/null
  if [ $? -ne 0 ]
  then
  	BOOL=0
  fi
  iput $IRODS_RESOURCE -f "$PATH_TEMP/$SPECIAL" $SUBDIR >/dev/null
  if [ $? -ne 0 ]
  then
	  BOOL=0
  fi
  # Removes the created files on the client
	rm $HELP
	if [ $? -ne 0 ]
	then
		BOOL=0
		echo "Error: Can't delete temporary files\n" 1>&2
	fi   
else
	echo "Files already created."
fi


# If no error happened, the measurement begins
if [ $BOOL -ne 1 ] 
then
	echo "Error: No files created\n" 1>&2
else
	echo "All files created."
  echo "Start to get files." 
 	#time: real_time;user_time;system_time
	$BENCHIT_TOOL_TIME -f "%e;%U;%S" -o "$BENCHIT_KERNEL_PATH_TEMP/$FILE" -a iget $IRODS_RESOURCE $IRODS_PROT $IRODS_THREADS -f "$SUBDIR/$SPECIAL" "$PATH_TEMP/$SPECIAL" >/dev/null
	echo "$1;$2" >> "$BENCHIT_KERNEL_PATH_TEMP/$FILE"
fi

# Clean up
if [ $BENCHIT_REMOVE_TEMP -eq 0 -o $3 -eq $BENCHIT_KERNEL_NUMBER_RUNS ]
then	 
	echo "Remove temporary files."
	rm -r $PATH_TEMP
	irm -fr $SUBDIR 2>/dev/null
	if [ $? -ne 0 ]
	then
		sleep 3
		irm -fr $SUBDIR
	fi

	if [ $3 -eq $BENCHIT_KERNEL_NUMBER_RUNS -a $BENCHIT_REMOVE_TEMP -ne 0 ]
	then
		echo "Last run. Files removed.\n"
	fi
else
	rm $HELP
fi


wait
