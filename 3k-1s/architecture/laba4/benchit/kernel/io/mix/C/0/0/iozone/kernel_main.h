/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/mix/C/0/0/iozone/kernel_main.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/


#ifndef __work_h
#define __work_h

#define STRINGSIZE 512

typedef struct mydata{
		char * filename;
    char * filesize_min;
    char * filesize_min_unit;
    char * filesize_max;
    char * filesize_max_unit;
    char * filesize_inc;
    char * filesize_inc_unit;
    char * recordsize;
    char * testlist;
    char * cachelinesize;
    char * cachesize;
    char * options;
//    char ** testnamearray;
    unsigned int * testarray;
    unsigned int nr_tests; /* number of iozone tests */
    unsigned int numfunctions; /* many tests include 2 functions */
    unsigned long min;
    unsigned long max;
    unsigned long inc;
} mydata_t;

extern void work( char * );
#endif


