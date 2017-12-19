/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: iobigread.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/read/C/PThread/0/big_files/iobigread.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#ifndef iobigread_h
#define iobigread_h

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

typedef struct io_data_struct
        {
	double FILESIZE;
	double DISKSPACE;
	double RAMSIZE;

	int POTFILPERDIR;
	int FILESPERDIR;
	int FILESPERTHREAD;
	int MAXREPEAT;
	int REPEATSTOP;
	int NUMCHANNELS;
	int CHANNELFACTOR;
	int TIMELIMIT;

	char * DISKPATH; 
	char * TMPHEADER;

        char *path;
	char *startpath;
	long maxdeep;
        } iods;

/*
    wrapperstruct for
    readfiles(problemsize, global->maxdeep, btime+i, etime+i);	    
*/
typedef struct 
	{
	iods *global;
	long problemsize;
	double *btime;
	double *etime;
	} thread_arg_wrapper_t;

extern void thread_readfile(thread_arg_wrapper_t *taw);

#endif /* #ifndef iobigread_h */


