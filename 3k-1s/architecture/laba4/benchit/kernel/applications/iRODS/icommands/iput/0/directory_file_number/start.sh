#! /bin/sh

# $1 = Increment of the number of files
# $2 = Increment-value of the last run 
# $3 = Number of runs

BENCHIT_KERNEL_NUMBER_FILES=$1
FILE=$BENCHIT_RESULT_NAME
SUBDIR="benchit_subdir_test_$1"
SUBDIR_TEMP="benchit_subdir_test_$2"
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

PATH_TEMP="$PATH_TEMP/$SUBDIR"
# Creates new files if they not already exist

if [ ! -d $PATH_TEMP ]
then
	mkdir $PATH_TEMP
	for i in `seq 1 $BENCHIT_KERNEL_NUMBER_FILES` 
	do 
    dd if=/dev/urandom of="$PATH_TEMP/$i$SPECIAL" bs="$BENCHIT_KERNEL_FILE_BLOCK_SIZE$BENCHIT_KERNEL_FILE_UNIT" count=$BENCHIT_KERNEL_FILE_BLOCK_NUMBER  2>/dev/null
    if [ $? -ne 0 ]
    then
      BOOL=0
    fi     
	done
else
	echo "Files already created."	
fi

# If no error happened, the measurement begins
if [ $BOOL -ne 1 ] 
then
  echo "Error: No files created\n" 1>&2
else
  echo "All files created."
  echo "Start to copy files."
 	#time: real_time;user_time;system_time
	$BENCHIT_TOOL_TIME -f "%e;%U;%S" -o "$BENCHIT_KERNEL_PATH_TEMP/$FILE" -a iput $IRODS_RESOURCE $IRODS_PROT $IRODS_THREADS -fr $PATH_TEMP ./ >/dev/null
	echo "$1" >> "$BENCHIT_KERNEL_PATH_TEMP/$FILE"
fi

# Clean up
if [ $BENCHIT_REMOVE_TEMP -eq 0 -o $3 -eq $BENCHIT_KERNEL_NUMBER_RUNS ]
then	
	echo "Remove temporary files."
  rm -r "$PATH_TEMP" 	
	if [ $3 -eq $BENCHIT_KERNEL_NUMBER_RUNS -a $BENCHIT_REMOVE_TEMP -ne 0 ]
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
