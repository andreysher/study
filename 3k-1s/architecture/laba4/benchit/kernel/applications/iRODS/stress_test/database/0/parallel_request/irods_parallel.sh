#! /bin/sh

# $1 = Filename
# $2 = Number of Processes
# $3 = Number of loops

TEST ()
{
	for i in `seq 1 $2`; do 
		imeta add -d $1 "benchit_$i_$3" "benchit"
	done
}
for i in `seq 1 $2`; do 
 	TEST "$1" "$3" "$i" &
done
wait
