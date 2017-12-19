/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: iobig_writefct.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/read/C/PThread/0/big_files/iobig_writefct.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#ifndef iobig_writefct_h
#define iobig_writefct_h
char *inttobinstring(long number);
void CreateFile(char *buffer, char* filename);
int CreateTree(long maxsub, char *path, long maxdeep, long deep, long *actual, char *buffer, long maxfiles);
void ioCreate(char *buffer, long number);
#endif


