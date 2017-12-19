/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: io_ls_types.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/ls/C/0/0/io_ls/io_ls_types.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "io_ls_types.h"

dir_tree_descr_t init_tree_descr (const char* root, int depth) {
    dir_tree_descr_t tmp;
    if (root == NULL) {
	fprintf (stderr, "init_tree_descr: root pointer is NULL (undefined environment .. ?)\n");
	exit (127); }
    tmp= (dir_tree_descr_t) malloc (sizeof (struct str_dir_tree_descr_t));
    tmp->root= (char *) malloc (strlen (root) + 1);
    strcpy (tmp->root, root); tmp->depth= depth;
    return tmp;
}

void free_tree_descr (dir_tree_descr_t d) {
    free (d->root); free (d);
}



