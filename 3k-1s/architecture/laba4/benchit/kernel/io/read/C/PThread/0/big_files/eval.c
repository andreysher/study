/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: eval.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/read/C/PThread/0/big_files/eval.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Reads the environment variables used by this kernel. */
/* may change to int-return-value (number of errors). */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "iobigread.h"
#include "interface.h"


void evaluate_environment(iods * pmydata)
{
   char * p = 0;
/*
pmydata->FILESIZE = (double)atof((const char *)bi_getenv("BENCHIT_KERNEL_FILESIZE",0));
pmydata->POTFILPERDIR = (int)atoi((const char *)bi_getenv("BENCHIT_KERNEL_POTFILPERDIR",0));
pmydata->FILESPERDIR = (int)atoi((const char *)bi_getenv("BENCHIT_KERNEL_FILESPERDIR",0));
pmydata->FILESPERTHREAD = (int)atoi((const char *)bi_getenv("BENCHIT_KERNEL_FILESPERTHREAD",0));
pmydata->MAXREPEAT = (int)atoi((const char *)bi_getenv("BENCHIT_KERNEL_MAXREPEAT",0));
pmydata->REPEATSTOP = (int)atoi((const char *)bi_getenv("BENCHIT_KERNEL_REPEATSTOP",0));

pmydata->DISKPATH = (char *) malloc(sizeof(char) * 128);
strncpy(pmydata->DISKPATH, (const char *) bi_getenv("BENCHIT_KERNEL_DISKPATH",0), 128);

pmydata->TMPHEADER = (char *) malloc(sizeof(char) * 128);
strncpy(pmydata->TMPHEADER, (const char *) bi_getenv("BENCHIT_KERNEL_TMPHEADER",0), 128);

pmydata->DISKSPACE = (double)atof((const char *)bi_getenv("BENCHIT_KERNEL_DISKSPACE",0));
pmydata->NUMCHANNELS = (int)atoi((const char *)bi_getenv("BENCHIT_KERNEL_NUMCHANNELS",0));
pmydata->CHANNELFACTOR = (int)atoi((const char *)bi_getenv("BENCHIT_KERNEL_CHANNELFACTOR",0));
pmydata->RAMSIZE = (double)atof((const char *)bi_getenv("BENCHIT_KERNEL_RAMSIZE",0));
pmydata->TIMELIMIT = (int)atoi((const char *)bi_getenv("BENCHIT_KERNEL_TIMELIMIT",0));
*/

pmydata->FILESIZE = (double)atof((const char *)getenv("BENCHIT_KERNEL_FILESIZE"));
pmydata->POTFILPERDIR = (int)atoi((const char *)getenv("BENCHIT_KERNEL_POTFILPERDIR"));
pmydata->FILESPERDIR = (int)atoi((const char *)getenv("BENCHIT_KERNEL_FILESPERDIR"));
pmydata->FILESPERTHREAD = (int)atoi((const char *)getenv("BENCHIT_KERNEL_FILESPERTHREAD"));
pmydata->MAXREPEAT = (int)atoi((const char *)getenv("BENCHIT_KERNEL_MAXREPEAT"));
pmydata->REPEATSTOP = (int)atoi((const char *)getenv("BENCHIT_KERNEL_REPEATSTOP"));

pmydata->DISKPATH = (char *) malloc(sizeof(char) * 128);
strncpy(pmydata->DISKPATH, (const char *) getenv("BENCHIT_KERNEL_DISKPATH"), 128);

pmydata->TMPHEADER = (char *) malloc(sizeof(char) * 128);
strncpy(pmydata->TMPHEADER, (const char *) getenv("BENCHIT_KERNEL_TMPHEADER"), 128);

pmydata->DISKSPACE = (double)atof((const char *)getenv("BENCHIT_KERNEL_DISKSPACE"));
pmydata->NUMCHANNELS = (int)atoi((const char *)getenv("BENCHIT_KERNEL_NUMCHANNELS"));
pmydata->CHANNELFACTOR = (int)atoi((const char *)getenv("BENCHIT_KERNEL_CHANNELFACTOR"));
pmydata->RAMSIZE = (double)atof((const char *)getenv("BENCHIT_KERNEL_RAMSIZE"));
/*
 * pmydata->TIMELIMIT = (int)atoi((const char *)getenv("BENCHIT_KERNEL_TIMELIMIT"));
 */
}


