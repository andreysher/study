#! /bin/sh

# $1 = Filename
# $2 = Number of metadata
# $3 = BENCHIT_KERNEL_META_ATTRIBUTE

if [ $3 -eq 0 ]
then
	for i in `seq 1 $2`; do 
    imeta add -d $1 "benchit" "benchit_$i" &
	done
else
	for i in `seq 1 $2`; do 
    imeta add -d $1 "benchit_$i" "benchit" &
	done
fi

wait
