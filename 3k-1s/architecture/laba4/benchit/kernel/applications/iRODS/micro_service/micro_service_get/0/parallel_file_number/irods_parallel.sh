#! /bin/sh

# $1 = Dateiname
# $2 = Number of files
# $3 = iRods collection
# $4 = Path of the file with the rule

for i in `seq 1 $2`; do 
	irule -F $4 "./$i$1" "$3/$i$1" &
done

wait
