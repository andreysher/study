/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: read_dirs.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/ls/C/0/0/io_ls/read_dirs.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "read_dirs.h"


int read_dir(const char* dir) {
    DIR     *dp;
    int     files;

    files= 0;
    dp = opendir(dir);                 
    if (!dp) return -1; /* could not opened */
    while (readdir (dp)) {
	files++; }
    closedir(dp);
    /* printf("%i files in %s\n", files, dir); */
    return files;
}

/* Same with output (for the sake of completeness
int read_dir(const char* dir) {
    DIR            *dp;
    int            files;
    struct dirent  *dirp;

    files= 0;
    dp = opendir(dir);                 
    while ((dirp = readdir (dp))) {
	printf ("%s\n", dirp->d_name);
	files++; 
    }
    closedir(dp);
    return files;
}
*/
 
int intern_read_dirs (const dir_tree_descr_t tree, int myDepth, int numberDirs, int randomly, int overhead) {
    int   number, i, j, files, dfiles;
    char  dirName[PATH_NAME_SIZE], pathPrefix[PATH_NAME_SIZE];

    if (myDepth > tree->depth) {
	fprintf (stderr, "read_dirs called with depth %i. It is reduced to depth of tree (%i)\n.", 
		 myDepth, tree->depth);
	myDepth= tree->depth; }
    strcpy (pathPrefix, tree->root); strcat (pathPrefix, "/d");
    for (i= files= 0, number= 1 << myDepth, j= rand() % number; i < numberDirs; i++, j++) {
	binary2name ((randomly ? rand() : j) % number, myDepth, dirName, PATH_NAME_SIZE, pathPrefix, "/d", "");
	/* for overhead determination do not read the directories */
	if (!overhead) {
	    dfiles= read_dir(dirName);
	    if (dfiles < 0) return -1;
	    files+= dfiles; }
    }
    return files;
}

int read_dirs_regularly (const dir_tree_descr_t tree, int myDepth, int numberDirs) {
    return intern_read_dirs (tree, myDepth, numberDirs, 0, 0);
}

int read_dirs_randomly (const dir_tree_descr_t tree, int myDepth, int numberDirs) {
    return intern_read_dirs (tree, myDepth, numberDirs, 1, 0);
}

int overhead_read_dirs_regularly (const dir_tree_descr_t tree, int myDepth, int numberDirs) {
    return intern_read_dirs (tree, myDepth, numberDirs, 0, 1);
}

int overhead_read_dirs_randomly (const dir_tree_descr_t tree, int myDepth, int numberDirs) {
    return intern_read_dirs (tree, myDepth, numberDirs, 1, 1);
}


