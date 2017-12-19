/*** Copyright (c), The Regents of the University of California            ***
 *** For more information please refer to files in the COPYRIGHT directory ***/
/* 
 * iget - The irods get utility
*/

#include <sys/time.h>
#include <libgen.h>

#include "rodsClient.h"
#include "parseCommandLine.h"
#include "rodsPath.h"
#include "getUtil.h"
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
   
    optStr = "hfKN:n:rQvVX:R:T";
   
    status = parseCmdLineOpt (argc, argv, optStr, 0, &myRodsArgs);

    if (status < 0) {
        printf("Use -h for help.\n");
        exit (1);
    }
    if (myRodsArgs.help==True) {
       usage();
       exit(0);
    }

    gettimeofday(zeit + 2,NULL);
    status = getRodsEnv (&myEnv);

    if (status < 0) {
        rodsLogError(LOG_ERROR, status, "main: getRodsEnv error. ");
        exit (1);
    }
   gettimeofday(zeit + 3,NULL);
    status = parseCmdLinePath (argc, argv, optind, &myEnv,
      UNKNOWN_OBJ_T, UNKNOWN_FILE_T, 0, &rodsPathInp);

    if (status < 0) {
        rodsLogError (LOG_ERROR, status, "main: parseCmdLinePath error. ");
        printf("Use -h for help.\n");
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
   gettimeofday(zeit + 5,NULL);
    if (conn == NULL) {
        exit (2);
    }
   gettimeofday(zeit + 6,NULL);
    if (strcmp (myEnv.rodsUserName, PUBLIC_USER_NAME) != 0) { 
        status = clientLogin(conn);
        if (status != 0) {
            rcDisconnect(conn);
            exit (7);
	}
    }
   gettimeofday(zeit + 7,NULL);
   gettimeofday(zeit + 8,NULL);
    status = getUtil (conn, &myEnv, &myRodsArgs, &rodsPathInp);
   gettimeofday(zeit + 9,NULL);
    rcDisconnect(conn);
   gettimeofday(zeit + 1,NULL);
    if (status < 0) {
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
usage () {
   char *msgs[]={
"Usage: iget [-fKQrUvVT] [-n replNumber] [-N numThreads] [-X restartFile]",
"[-R resource] srcDataObj|srcCollection ... destLocalFile|destLocalDir",
"Usage : iget [-fKQUvVT] [-n replNumber] [-N numThreads] [-X restartFile]",
"[-R resource] srcDataObj|srcCollection",
"Usage : iget [-fKQUvVT] [-n replNumber] [-N numThreads] [-X restartFile]",
"[-R resource] srcDataObj ... -",
"Get data-objects or collections from irods space, either to the specified",
"local area or to the current working directory.",
" ",
"If the destLocalFile is '-', the files read from the server will be ",
"written to the standard output (stdout). Similar to the UNIX 'cat'",
"command, multiple source files can be specified.",
" ",
"The -X option specifies that the restart option is on and the restartFile",
"input specifies a local file that contains the restart info. If the ",
"restartFile does not exist, it will be created and used for recording ",
"subsequent restart info. If it exists and is not empty, the restart info",
"contained in this file will be used for restarting the operation.",
"Note that the restart operation only works for uploading directories and",
"the path input must be identical to the one that generated the restart file",
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

" -f  force - write local files even it they exist already (overwrite them)",
" -K  verify the checksum",
" -n  replNumber - retrieve the copy with the specified replica number ",
" -N  numThreads - the number of thread to use for the transfer. A value of",
"       0 means no threading. By default (-N option not used) the server ",
"       decides the number of threads to use.", 
" -r  recursive - retrieve subcollections",
" -R  resource - the preferred resource",
" -T  renew socket connection after 10 minutes",
" -Q  use RBUDP (datagram) protocol for the data transfer",
" -v  verbose",
" -V  Very verbose",
"     restartFile input specifies a local file that contains the restart info.",
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
