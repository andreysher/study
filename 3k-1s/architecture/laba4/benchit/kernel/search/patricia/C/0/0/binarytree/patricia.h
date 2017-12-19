/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: patricia.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/search/patricia/C/0/0/binarytree/patricia.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Execution time of Insertion / Searching in a patricia-trie
 *         PATRICIA: Practical Algorithm To Retrieve Information
 *                   Coded In Alphanumeric
 *******************************************************************/

#ifndef __patricia_h
#define __patricia_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "interface.h"

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

/* length of the used keys in byte */
#define KEYLENGTH 16384

/* The data structure that holds the patricia-trie. */
typedef struct patricia_struct {
   unsigned char uckey[KEYLENGTH];
   int ibit;
   struct patricia_struct *pleft, *pright;
} patriciastruct;

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata {
   /* array for the created keys */
   unsigned char **ppuckeys;
   /* additional parameters */
   myinttype maxsize;
} mydata_t;

/*prototypes for patricia_funcs.c*/
extern int patriciacompare(unsigned char *pucelement1,
                           unsigned char *pucelement2);
extern patriciastruct *patriciainit(void);
extern patriciastruct *patriciasearch(unsigned char *pucsearchkey,
                                      patriciastruct * pkeys);
extern int patriciainsert(unsigned char *pucinskey, patriciastruct * pkeys);
extern void patriciafree(patriciastruct * pkeys);

#endif

