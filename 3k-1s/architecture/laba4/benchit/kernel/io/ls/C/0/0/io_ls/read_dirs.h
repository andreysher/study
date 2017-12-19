/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: read_dirs.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/ls/C/0/0/io_ls/read_dirs.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#ifndef BENCHIT_READ_DIRS_H
#define BENCHIT_READ_DIRS_H

#include "create_dirs.h"


/** Reads silently the content of a directory and returns the number of files (-1 if failed) */
int read_dir(const char* dir); 

/** Reads silently the content of directory tree at depth <myDepth> (numberDirs times, if necessary restarts) 
 *  and returns the number of files (-1 if failed) */
int read_dirs_regularly(const dir_tree_descr_t tree, int myDepth, int numberDirs);

/** Reads silently the content of directory tree at depth <myDepth> in random order (numberDirs times)
 *  and returns the number of files (-1 if failed) */
int read_dirs_randomly(const dir_tree_descr_t tree, int myDepth, int numberDirs);

/** Overhead of read_dirs_regularly */
int overhead_read_dirs_regularly(const dir_tree_descr_t tree, int myDepth, int numberDirs);

/** Overhead of read_dirs_randomly */
int overhead_read_dirs_randomly(const dir_tree_descr_t tree, int myDepth, int numberDirs);

/*  Internal function used by the 4 above and by io_ls because of easier handling */
int intern_read_dirs (const dir_tree_descr_t tree, int myDepth, int numberDirs, int randomly, int overhead);

#endif /* BENCHIT_READ_DIRS_H */

