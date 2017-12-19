/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: io_ls.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/ls/C/0/0/io_ls/io_ls.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#ifndef __io_ls_h
#define __io_ls_h


#include "io_ls_types.h"
#include "read_dirs.h"


#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

typedef struct  {
    dir_tree_descr_t  myTree;
    int               filesPerDir;
    int               dirsToRead;
    int               steps;
    
} mydata_t;

#endif /* __io_ls_h */
