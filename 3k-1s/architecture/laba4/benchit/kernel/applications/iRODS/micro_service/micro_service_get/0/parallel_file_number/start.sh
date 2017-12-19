#! /bin/sh

# $1 = Increment of the number of files
# $2 =  Increment-value of the last run 
# $3 = Number of runs

BENCHIT_KERNEL_NUMBER_FILES=$1
FILE=$BENCHIT_RESULT_NAME
SPECIAL="_benchit"
SUBDIR="benchit_subdir"
PATH_TEMP=$BENCHIT_KERNEL_PATH_TEMP
PATH_SCRIPT=$KERNELDIR
IRODS_RESOURCE=""
PATH_NOW=`pwd`
BOOL=1

if [ "$BENCHIT_IRODS_RESC" != "" ]
then
	IRODS_RESOURCE="-R $BENCHIT_IRODS_RESC"
fi


PATH_TEMP="$PATH_TEMP/$SUBDIR/"
HELP="$PATH_TEMP*"

# Creates new files if they not already exist and transfers them to iRODS
if [ ! -d $PATH_TEMP ]
then
	mkdir $PATH_TEMP
	imkdir $SUBDIR

	for i in `seq 1 $BENCHIT_KERNEL_NUMBER_FILES` 
	do 
    dd if=/dev/urandom of="$PATH_TEMP/$i$SPECIAL" bs="$BENCHIT_KERNEL_FILE_BLOCK_SIZE$BENCHIT_KERNEL_FILE_UNIT" count=$BENCHIT_KERNEL_FILE_BLOCK_NUMBER  2>/dev/null
    if [ $? -ne 0 ]
    then
      BOOL=0
    fi     
	done

  iput $IRODS_RESOURCE -fr $HELP $SUBDIR >/dev/null
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
   echo "Error: Can't execute the measurement (no files created)\n" 1>&2
else
	echo "All files created."
  echo "Start to get files."
  cd $PATH_TEMP
 	#time: real_time;user_time;system_time
	$BENCHIT_TOOL_TIME -f "%e;%U;%S" -o "$BENCHIT_KERNEL_PATH_TEMP/$FILE" -a $PATH_SCRIPT/irods_parallel.sh "$SPECIAL" "$BENCHIT_KERNEL_NUMBER_FILES" "$SUBDIR" "$BENCHIT_SPEZIAL_IR"
	echo "$1" >> "$BENCHIT_KERNEL_PATH_TEMP/$FILE"
fi
cd $PATH_NOW

# Clean up
if [ $BENCHIT_REMOVE_TEMP -eq 0 -o $3 -eq $BENCHIT_KERNEL_NUMBER_RUNS ]
then	
	echo "Remove temporary files."
  rm -r "$PATH_TEMP" 
	
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
