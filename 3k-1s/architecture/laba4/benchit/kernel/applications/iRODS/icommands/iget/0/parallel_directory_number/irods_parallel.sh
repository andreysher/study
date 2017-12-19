#! /bin/sh

# $1 = Path of the files
# $2 = Number of directories
# $3 = iRODS resource
# $4 = iRODS number of threads
# $5 = iRODS protocol
# $6 = iRODS destination Collection
# $7 = Temporary directory

for i in `seq 1 $2`; do 
	iget $3 $4 $5 -fr "./$6/$7$i" "$1"  &
done

wait
