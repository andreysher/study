/*** Copyright (c), The Regents of the University of California            ***
 *** For more information please refer to files in the COPYRIGHT directory ***/
/* 
 * iput - The irods put utility
*/

#include <sys/time.h> 
#include <libgen.h>

#include "rodsClient.h"
#include "parseCommandLine.h"
#include "rodsPath.h"
#include "putUtil.h"
void usage ();

int
main(int argc, char **argv) {
    int status;
    rodsEnv myEnv;
    rErrMsg_t errMsg;
    rcComm_t *conn;
    rodsArguments_t myRodsArgs;
    char *optStr;
    rodsPathInp_t rodsPathInp;
    int reconnFlag;
    
   struct timeval zeit[10];
   gettimeofday(zeit + 0,NULL);
   
    optStr = "aD:fhkKn:N:p:rR:QTvVX:";
   
    status = parseCmdLineOpt (argc, argv, optStr, 0, &myRodsArgs);

    if (status < 0) {
	printf("use -h for help.\n");
        exit (1);
    }

    if (myRodsArgs.help==True) {
       usage();
       exit(0);
    }

    gettimeofday(zeit + 2,NULL);
    status = getRodsEnv (&myEnv);
    if (status < 0) {
        rodsLogError (LOG_ERROR, status, "main: getRodsEnv error. ");
        exit (1);
    }
    gettimeofday(zeit + 3,NULL);

    status = parseCmdLinePath (argc, argv, optind, &myEnv,
      UNKNOWN_FILE_T, UNKNOWN_OBJ_T, 0, &rodsPathInp);

    if (status < 0) {
        rodsLogError (LOG_ERROR, status, "main: parseCmdLinePath error. "); 
	printf("use -h for help.\n");
        exit (1);
    }

    gettimeofday(zeit + 4,NULL);
    if (myRodsArgs.reconnect == True) {
        reconnFlag = RECONN_TIMEOUT;
    } else {
        reconnFlag = NO_RECONN;
    }
    
    conn = rcConnect (myEnv.rodsHost, myEnv.rodsPort, myEnv.rodsUserName,
      myEnv.rodsZone, reconnFlag, &errMsg);
    
    if (conn == NULL) {
        exit (2);
    }
   gettimeofday(zeit + 5,NULL);
   
    gettimeofday(zeit + 6,NULL);
    status = clientLogin(conn);
    if (status != 0) {
       rcDisconnect(conn);
        exit (7);
    }
   gettimeofday(zeit + 7,NULL);
   gettimeofday(zeit + 8,NULL);
    status = putUtil (conn, &myEnv, &myRodsArgs, &rodsPathInp);
   gettimeofday(zeit + 9,NULL);
    rcDisconnect(conn);
   gettimeofday(zeit + 1,NULL);
    if (status < 0) 
    {
	   exit (3);
    } 
    else 
    {
      double start,stop;
      int i,help;
      //printf("Datei: %s \n",rodsPathInp.targPath->outPath);
      char file_name[20];
      char file_help[20];
      strncpy(file_name,basename(rodsPathInp.targPath->outPath),20);
      strncpy(file_help,file_name,20);
      char *str_help;
      str_help = strchr(file_help,'_');
      if (str_help != 0)
         *str_help = '\0';

      for(i = 0 ; i < 5 ; i++)
      {
         help = i * 2;
         start = (double) zeit[help].tv_sec + ((double) zeit[help].tv_usec / 1000000.0);
         stop  = (double) zeit[help + 1].tv_sec + ((double) zeit[help + 1].tv_usec / 1000000.0);
         switch(i)
         {
            case 0 : printf("%s;%f;",file_help,stop - start); break;
            case 1 : printf("%f;",stop - start); break;
            case 2 : printf("%f;",stop - start); break;
            case 3 : printf("%f;",stop - start); break;
            case 4 : printf("%f\n",stop - start); break;
            default: printf("%s:Error\n",file_name);
         }
      }
      
        exit(0);
    }

}

void 
usage ()
{
   char *msgs[]={
"Usage : iput [-fkKQrTUvV] [-D dataType] [-N numThreads] [-n replNum]",
"             [-p physicalPath] [-R resource] [-X restartFile]", 
"		localSrcFile|localSrcDir ...  destDataObj|destColl",
"Usage : iput [-fkKQTUvV] [-D dataType] [-N numThreads] [-n replNum] ",
"             [-p physicalPath] [-R resource] [-X restartFile] localSrcFile",
" ",
"Store a file into iRODS.  If the destination data-object or collection are",
"not provided, the current irods directory and the input file name are used.",
"The -X option specifies that the restart option is on and the restartFile",
"input specifies a local file that contains the restart info. If the ",
"restartFile does not exist, it will be created and used for recording ",
"subsequent restart info. If it exists and is not empty, the restart info",
"contained in this file will be used for restarting the operation.",
"Note that the restart operation only works for uploading directories and",
"the path input must be identical to the one that generated the restart file", 
" ",
"If the options -f is used to overwrite an existing data-object, the copy",
"in the resource specified by the -R option will be picked if it exists.",
"Otherwise, one of the copy in the other resources will be picked for the",
"overwrite. Note that a copy will not be made in the specified resource",
"if a copy in the specified resource does not already exist. The irepl",
"command should be used to make a replica of an existing copy.", 
" ",
"The -Q option specifies the use of the RBUDP transfer mechanism which uses",
"the UDP protocol for data transfer. The UDP protocol is very efficient",
"if the network is very robust with few packet losses. Two environment",
"variables - rbudpSendRate and rbudpPackSize are used to tune the RBUDP",
"data transfer. rbudpSendRate is used to throttle the send rate in ",
"kbits/sec. The default rbudpSendRate is 600,000. rbudpPackSize is used",
"to set the packet size. The dafault rbudpPackSize is 8192.",
" ",
"The -T option will renew the socket connection between the client and ",
"server after 10 minutes of connection. This gets around the problem of",
"sockets getting timed out by the firewall as reported by some users.",
" ",
"Options are:",
" -D  dataType - the data type string",
" -f  force - write data-object even it exists already; overwrite it",
" -k  checksum - calculate a checksum on the data",
" -K  verify checksum - calculate and verify the checksum on the data",
" -N  numThreads - the number of thread to use for the transfer. A value of",
"       0 means no threading. By default (-N option not used) the server ",
"       decides the number of threads to use.",
" -Q  use RBUDP (datagram) protocol for the data transfer",
" -R  resource - specifies the resource to store to. This can also be specified",
"     in your environment or via a rule set up by the administrator.",
" -r  recursive - store the whole subdirectory",
" -T  renew socket connection after 10 minutes",
" -v  verbose",
" -V  Very verbose",
" -X  restartFile - specifies that the restart option is on and the",
"     restartFile input specifies a local file that contains the restart info.",

" -h  this help",
""};
   int i;
   for (i=0;;i++) {
      if (strlen(msgs[i])==0) return;
      printf("%s\n",msgs[i]);
   }
}
