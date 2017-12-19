#! /bin/sh

# $1 = Number of directories
# $2 = Increment-value of the last run 
# $3 = Number of runs
# $4 = Number of files per directory
# $5 = Number of directories with one file more than the others

FILE=$BENCHIT_RESULT_NAME
SUBDIR="benchit_subdir"
SUBDIR_TEMP="subdir"
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
	for i in `seq 1 $1` 
	do 
    PATH_HELP="$PATH_TEMP/$SUBDIR_TEMP$i"
    mkdir $PATH_HELP
    FILES_TEMP=$4
    if [ $i -le $5 ]
    then
    	FILES_TEMP=$(($FILES_TEMP+1))
    fi
    for j in `seq 1 $FILES_TEMP`
    do
    	dd if=/dev/urandom of="$PATH_HELP/$j$SPECIAL" bs="$BENCHIT_KERNEL_FILE_BLOCK_SIZE$BENCHIT_KERNEL_FILE_UNIT" count=$BENCHIT_KERNEL_FILE_BLOCK_NUMBER  2>/dev/null
    	if [ $? -ne 0 ]
    	then
      	BOOL=0
    	fi     
		done
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
 	#time: real_time;user_time;system_time
	$BENCHIT_TOOL_TIME -f "%e;%U;%S" -o "$BENCHIT_KERNEL_PATH_TEMP/$FILE" -a $PATH_SCRIPT/irods_parallel.sh "$PATH_TEMP" "$1" "$IRODS_RESOURCE" "$IRODS_PROT" "$IRODS_THREADS" "$SUBDIR" "$SUBDIR_TEMP">/dev/null
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

for i in `seq 1 $1`; do 
	irm -fr "$SUBDIR/$SUBDIR_TEMP$i" &
done

wait

irm -fr $SUBDIR 2>/dev/null
if [ $? -ne 0 ]
then
	sleep 3
	irm -fr $SUBDIR
fi 


wait
