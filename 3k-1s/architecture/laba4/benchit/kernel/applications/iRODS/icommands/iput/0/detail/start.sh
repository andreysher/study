#! /bin/sh
# $1 = number of runs

FILE=$BENCHIT_RESULT_NAME
FILE_2=$BENCHIT_RESULT_NAME"_extra"
SUBDIR="benchit_subdir"
SPECIAL="_benchit"
PATH_TEMP=$BENCHIT_KERNEL_PATH_TEMP
PATH_SCRIPT=$KERNELDIR
IRODS_RESOURCE=""
IRODS_PROT=""
IRODS_THREADS=""
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
PATH_TEMP="$PATH_TEMP/$SUBDIR/"
# Creates new files if they not already exist
if [ ! -d $PATH_TEMP ]
then
	mkdir $PATH_TEMP
	
	for i in `seq 1 $BENCHIT_KERNEL_NUMBER_FILES` 
	do 
  	dd if=/dev/urandom of="$PATH_TEMP/$i$SPECIAL" bs="$BENCHIT_KERNEL_FILE_BLOCK_SIZE$BENCHIT_KERNEL_FILE_UNIT" count=$BENCHIT_KERNEL_FILE_BLOCK_NUMBER 2>/dev/null
    if [ $? -ne 0 ]
    then
      BOOL=0
    fi     
	done
else
	echo "Files already created.\n"	
fi

# If no error happened, the measurement begins
if [ $BOOL -ne 1 ] 
then
	echo "Error: No files created\n" 1>&2
else
  imkdir $SUBDIR
  echo "All files created."
  echo "Start to copy files."
  #time: real_time;user_time;system_time
  $PATH_SCRIPT/irods_parallel.sh "$SPECIAL" "$PATH_TEMP" "$BENCHIT_KERNEL_NUMBER_FILES" "$IRODS_RESOURCE" "$IRODS_PROT" "$IRODS_THREADS" "$SUBDIR">> "$BENCHIT_KERNEL_PATH_TEMP/$FILE"
fi

# Clean up
if [ $BENCHIT_REMOVE_TEMP -eq 0 -o $1 -eq $BENCHIT_KERNEL_NUMBER_RUNS ]
then
	echo "Remove temporary files."
	rm -r "$PATH_TEMP" 	
	if [ $1 -eq $BENCHIT_KERNEL_NUMBER_RUNS -a $BENCHIT_REMOVE_TEMP -ne 0 ]
	then
		echo "Last run. Files removed.\n"
  fi
fi

irm -fr $SUBDIR 2>/dev/null
if [ $? -ne 0 ]
then
	sleep 3
	irm -fr $SUBDIR
fi


wait
