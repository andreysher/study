#! /bin/sh

# $1 = Path of the iRODS-Directory

PATH="$1/clients/icommands/src"
MV="/bin/mv"
RM="/bin/rm"
TEST=0


echo "Search for iput.c.backup and iget.c.backup"
cd $PATH
if [ -f "iput.c.backup" -a -f "iput.c" ]
then
   $RM "iput.c"
   if [ $? -ne 0 ]
   then
      TEST=1
      echo "An error occurred while removing iput.c\n"
   fi
   $MV "iput.c.backup" "iput.c"
   if [ $? -ne 0 ]
   then
      TEST=1
      echo "An error occurred while renaming iput.c.backup\n"
   fi
else
   echo "Can't find iput.c or iput.c.backup\n"
fi
if [ -f "iget.c.backup" -a -f "iget.c" ]
then
   $RM "iget.c"
   if [ $? -ne 0 ]
   then
      TEST=1
      echo "An error occurred while removing iget.c\n"
   fi
   $MV "iget.c.backup" "iget.c"
   if [ $? -ne 0 ]
   then
      TEST=1
      echo "An error occurred while renaming iget.c.backup\n"
   fi
else
   echo "Can't find iget.c or iget.c.backup\n"
fi

if [ $TEST -eq 0 ]
then
   echo "All done\n"
fi

