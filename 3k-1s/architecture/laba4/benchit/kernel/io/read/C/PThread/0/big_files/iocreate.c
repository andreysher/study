/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: iocreate.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/read/C/PThread/0/big_files/iocreate.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "iobig_writefct.h"
#include "interface.h"
#include "iobigread.h"
#include "eval.h"

int createFiles()
{
	const char *path;
	char *start, *destination, *buffer;
	FILE *fp;
	long i;
	double ds, fs;
	double FILESIZE=0.0, DISKSPACE=0.0, RAMSIZE=0.0 ;

	int POTFILPERDIR=0, FILESPERDIR=0, FILESPERTHREAD=0, MAXREPEAT=0,
    REPEATSTOP=0, NUMCHANNELS=0, CHANNELFACTOR=0, TIMELIMIT=0;

	char * DISKPATH=NULL; 
	char * TMPHEADER=NULL;

	iods * pmydata;

	printf("\ncreate\n"); fflush(NULL);

	pmydata = (iods *) malloc(sizeof(iods));

	evaluate_environment(pmydata);

	printf("\nevaluate\n"); fflush(NULL);

	FILESIZE = pmydata->FILESIZE;
	POTFILPERDIR = pmydata->POTFILPERDIR ;
	FILESPERDIR =  pmydata->FILESPERDIR ;
	FILESPERTHREAD = pmydata->FILESPERTHREAD;
	MAXREPEAT = pmydata->MAXREPEAT;
	REPEATSTOP = pmydata->REPEATSTOP;
	DISKPATH = (char *) malloc(sizeof(char) * 128);
	DISKPATH = pmydata->DISKPATH ;
	TMPHEADER = (char *) malloc(sizeof(char) * 128);
	TMPHEADER = pmydata->TMPHEADER;
	DISKSPACE = pmydata->DISKSPACE;
	NUMCHANNELS = pmydata->NUMCHANNELS;
	CHANNELFACTOR = pmydata->CHANNELFACTOR;
	RAMSIZE = pmydata->RAMSIZE;
	TIMELIMIT = pmydata->TIMELIMIT;

	i=0;
	/*ds => number of files that have to be created*/
	
	/* ds=DISKSPACE/FILESIZE; */
	ds=DISKSPACE;
	fs=FILESIZE;
	ds=ds/fs;

	printf("PATH=%s",DISKPATH);

	start=malloc(4096*sizeof(char));
	if(start==NULL) { printf("\nCant get memory to save execution path!\n"); return 1; }
	getcwd(start, 4096);

	path=DISKPATH;
	if(path==NULL) { printf("\nCant get path for writing!\n"); return 1; }

	/*iobig.tmp is created to save datas for the read and the remove program*/	
	destination=malloc((strlen(path)+32)*sizeof(char));
	if(destination==NULL) { printf("\nCant get memory for creating tmp-file path!\n"); return 1; }
	sprintf(destination, "%siobig.tmp", path);

	/*if the file that should be created already exists => try another name*/
	while(fopen(destination, "r")!=NULL)
		{
		i++;
		sprintf(destination, "%siobig%ld.tmp", path, i);
		}
	
	printf("Creating temporary file %s\n", destination);
	fp=fopen(destination, "w");
	if(fp==NULL) { printf("\nCant create tmp-file! Maybe no write properties in provided path.\n"); return 1; }
	fprintf(fp, TMPHEADER); fprintf(fp, "\n");

	/*write indentify-string, created directory, filenumber and filesize to tmp-file*/
	sprintf(destination, "%stmp%ld/", path, (long)(time(NULL)));
	fprintf(fp, "%s", destination); fprintf(fp, "\n");
	fprintf(fp, "%f %f", ds, fs);
	fclose(fp);
	if(mkdir(destination, S_IRWXU)) { printf("\nCant create path: %s\n!", destination); return 1; }

	/*create the file that should be written*/
	buffer=malloc(FILESIZE);
        if(buffer==NULL) { printf("\nCant get memory for filecontent!\n"); return 1; }

	for(i=0;i<((long)(FILESIZE/sizeof(char)));i++) buffer[i]=(char)(rand()%255);

	if(chdir(destination)) { printf("\nCant change to directory %s\n", path); return 1; }

	/*create binary-tree with the files*/
	printf("Creating %ld files for the Benchmark\n", (long)(ds));
        ioCreate(buffer, (long)(ds-1));
	
	if(chdir(start)) printf("\nCant change to directory %s\n.", start);
	
	free(destination);
	free(start);
	return 0;
}
