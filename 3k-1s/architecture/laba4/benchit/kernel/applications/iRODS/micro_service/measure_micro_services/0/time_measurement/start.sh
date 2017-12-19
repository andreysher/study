#! /bin/sh

FILE_HELP="help_$BENCHIT_RESULT_NAME"

# Starts measurement
echo "Start to execute irule"
irule -F $BENCHIT_SPEZIAL_IR > "$BENCHIT_KERNEL_PATH_TEMP/$FILE_HELP";
# Filters the result 
sed -n '/benchit/p' "$BENCHIT_KERNEL_PATH_TEMP/$FILE_HELP" > $BENCHIT_SPEZIAL_RESULT

# Clean up
echo "Remove temporary files."
rm "$BENCHIT_KERNEL_PATH_TEMP//$FILE_HELP" &

wait
