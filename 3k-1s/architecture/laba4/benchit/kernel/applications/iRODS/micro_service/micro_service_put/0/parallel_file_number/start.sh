#! /bin/sh

# $1 = Increment of the number of files
# $2 = Increment-value of the last run 
# $3 = Number of runs

BENCHIT_KERNEL_NUMBER_FILES=$1
FILE=$BENCHIT_RESULT_NAME
SPECIAL="_benchit"
SUBDIR="benchit_subdir"
PATH_TEMP=$BENCHIT_KERNEL_PATH_TEMP
PATH_SCRIPT=$KERNELDIR
PATH_NOW=`pwd`
BOOL=1

PATH_TEMP="$PATH_TEMP/$SUBDIR/"
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
  imkdir $SUBDIR
  echo "All files created."
  echo "Start to copy files."
  cd $PATH_TEMP
 	#time: real_time;user_time;system_time
	$BENCHIT_TOOL_TIME -f "%e;%U;%S" -o "$BENCHIT_KERNEL_PATH_TEMP/$FILE" -a $PATH_SCRIPT/irods_parallel.sh "$SPECIAL" "$BENCHIT_KERNEL_NUMBER_FILES" "$BENCHIT_IRODS_RESC" "$BENCHIT_SPEZIAL_IR" "$SUBDIR"
 	echo "$1" >> "$BENCHIT_KERNEL_PATH_TEMP/$FILE"
fi
cd $PATH_NOW

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
