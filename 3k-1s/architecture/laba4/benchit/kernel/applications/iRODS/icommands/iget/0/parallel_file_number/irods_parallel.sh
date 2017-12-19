#! /bin/sh

# $1 = Filename
# $2 = Path of the files
# $3 = Number of files
# $4 = iRODS resource
# $5 = iRODS number of threads
# $6 = iRODS protocol
# $7 = iRODS collection

for i in `seq 1 $3`; do 
	iget $4 $5 $6 -f "$7/$i$1" "$2" &
done

wait
