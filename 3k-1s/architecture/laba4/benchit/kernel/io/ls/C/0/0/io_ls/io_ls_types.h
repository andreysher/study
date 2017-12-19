/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: io_ls_types.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/ls/C/0/0/io_ls/io_ls_types.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#ifndef BENCHIT_IO_LS_TYPES_H
#define BENCHIT_IO_LS_TYPES_H

#define STRING_SIZE 1024
#define PATH_NAME_SIZE STRING_SIZE

/* internal */
struct str_dir_tree_descr_t {
    char *root;
    int  depth;
};

/** Data structure to describe directory tree, contains root and depth of tree */
typedef struct str_dir_tree_descr_t* dir_tree_descr_t;


/** Initialize: allocate and copy root */
dir_tree_descr_t init_tree_descr (const char* root, int depth);

/** Set memory free */
void free_tree_descr (dir_tree_descr_t d); 

#endif /* BENCHIT_IO_LS_TYPES_H */


