#! /bin/sh

# $1 = Path of the iRODS-Directory

PATH="$1/clients/icommands/src"
PATH_TEMP=`pwd`
MV="/bin/mv"
CP="/bin/cp"
TEST=0

echo "Backup iput.c and iget.c\n"
cd $PATH
$MV "iput.c" "iput.c.backup"
if [ $? -ne 0 ]
then
   TEST=1
   echo "An error occurred while creating iput.c.backup\n"
fi
   
$MV "iget.c" "iget.c.backup"
if [ $? -ne 0 ]
then
   TEST=1
   echo "An error occurred while creating iget.c.backup\n"
fi
cd $PATH_TEMP
if [ $TEST -eq 0 ]
then
   echo "Copy new files iput.c and iget.c"
   $CP "iput.c" "iget.c" $PATH
   if [ $? -ne 0 ]
   then
      TEST=1
      echo "An error occurred while copying iput.c and iget.c\n"
   else
      echo "All done\n"
   fi
fi

