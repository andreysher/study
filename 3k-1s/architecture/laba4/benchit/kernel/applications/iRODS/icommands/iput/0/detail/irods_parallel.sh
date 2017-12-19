#! /bin/sh

# $1 = Filename
# $2 = Path of the files
# $3 = Number of files
# $4 = iRODS resource
# $5 = iRODS number of threads
# $6 = iRODS protocol
# $7 = iRODS Collection

for i in `seq 1 $3`; do 
	iput $4 $5 $6 -f "$2/$i$1" "$7" & 
done

wait
