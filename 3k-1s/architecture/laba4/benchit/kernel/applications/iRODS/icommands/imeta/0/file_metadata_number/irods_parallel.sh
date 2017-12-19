#! /bin/sh

# $1 = Filename

cat $1 | imeta 1>/dev/null

wait
