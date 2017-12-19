/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/mix/C/0/0/iozone_3d/kernel_main.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#ifndef __work_h
#define __work_h

#define STRINGSIZE 512

typedef struct mydata{
    char * filename;
    char * filesize_max;
    char * filesize_max_unit;
    char * cachelinesize;
    char * cachesize;
    char * options;
    unsigned long max;
    unsigned long listlength;
    unsigned long startline;
} mydata_t;

typedef struct list_node{
    unsigned long long data;
    struct list_node * next;
} list_t;

extern void work( char * );
#endif


