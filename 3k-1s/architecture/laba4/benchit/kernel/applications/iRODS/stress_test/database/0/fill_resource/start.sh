#! /bin/sh

# $1 = ID of the subdirectory

FILE=$BENCHIT_RESULT_NAME
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

# Creates new files
if [ ! -d $PATH_TEMP/$SUBDIR ]
then
	mkdir $PATH_TEMP/$SUBDIR
fi
SUBSUBDIR="$SUBDIR""_$1"
PATH_TEMP="$PATH_TEMP/$SUBDIR"
PATH_SUB="$PATH_TEMP/$SUBSUBDIR/"
mkdir $PATH_SUB
for i in `seq 1 $BENCHIT_KERNEL_FILES_NUMBER` 
do 
	dd if=/dev/urandom of="$PATH_SUB/$i$SPECIAL" bs="$BENCHIT_KERNEL_FILE_BLOCK_SIZE$BENCHIT_KERNEL_FILE_UNIT" count=$BENCHIT_KERNEL_FILE_BLOCK_NUMBER 2>/dev/null
  if [ $? -ne 0 ]
  then
  	BOOL=0
  fi     
done

# If no error happened, the measurement begins
if [ $BOOL -ne 1 ] 
then
  echo "Error: No files created\n" 1>&2
else
  echo "All files created."
  echo "Start to copy files."
 	#time: real_time;user_time;system_time
	$BENCHIT_TOOL_TIME -f "%e;%U;%S" -o "$BENCHIT_KERNEL_PATH_TEMP/$FILE" -a iput $IRODS_RESOURCE $IRODS_PROT $IRODS_THREADS -fr "$PATH_TEMP" ./ >/dev/null
	echo "$1" >> "$BENCHIT_KERNEL_PATH_TEMP/$FILE"
fi

# Clean up
echo "Remove temporary files."
rm -r $PATH_SUB 
if [ \( $BENCHIT_KERNEL_DELETE_FILES_IRODS -ne 0 \) -a \( $BENCHIT_KERNEL_FILL_NUMBER -eq $1 \) ]
then
	rm -r $PATH_TEMP 	
	irm -fr $SUBDIR 2>/dev/null
	if [ $? -ne 0 ]
	then
		sleep 3
		irm -fr $SUBDIR
	fi 
fi


wait
