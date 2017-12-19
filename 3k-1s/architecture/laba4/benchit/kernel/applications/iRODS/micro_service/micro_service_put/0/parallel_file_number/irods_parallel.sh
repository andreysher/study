#! /bin/sh

# $1 = Dateiname
# $2 = Number of files
# $3 = iRods - resource name
# $4 = Path of the file with the rule
# $5 = iRods collection

for i in `seq 1 $2`; do
	irule -F $4 "./$i$1" "$5/$i$1" $3 &
done

wait
