/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: patricia_funcs.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/search/patricia/C/0/0/binarytree/patricia_funcs.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Execution time of Insertion / Searching in a patricia-trie
 *         PATRICIA: Practical Algorithm To Retrieve Information
 *                   Coded In Alphanumeric
 *******************************************************************/

#include "patricia.h"

/****f* patricia_funcs.c/measurement::patriciacompare
 * SYNOPSIS
 * int patriciacompare(unsigned char *pucelement1, unsigned char *pucelement2)
 * DESCRIPTION
 * This function patriciacompare to char-arrays it returns 0
 * if they are not equal and 1 when they are equal.
 ***/
int patriciacompare(unsigned char *pucelement1, unsigned char *pucelement2) {
   /* variable for loops and buffering */
   int ii;

   IDL(1, printf("reached function patriciacompare\n"));

   /* compare each char value of the arrays */
   for (ii = 0; ii < KEYLENGTH; ii++) {
      /* if one is different ... */
      if (pucelement1[ii] != pucelement2[ii]) {
         IDL(1, printf("completed function patriciacompare\n"));
         /* ... comparing failed */
         return 0;
      }
   }
   IDL(1, printf("completed function patriciacompare\n"));

   /* if none is different -> comparing succeeded */
   return 1;
}

/****f* patricia_funcs.c/measurement::patriciainit
 * SYNOPSIS
 * patriciastruct *patriciainit()
 * DESCRIPTION
 * This function initializes a patricia trie.
 ***/
patriciastruct *patriciainit() {
   /* variable for loops and buffering */
   int ii;
   /* head of the initialized patricia trie -> return value of the function */
   patriciastruct *phead;

   /* allocate memory */
   phead = calloc(1, sizeof(patriciastruct));

   /* bit value of the head is maximum bit length */
   phead->ibit = KEYLENGTH * 8 - 1;
   /* left and right links are pointing to head itself */
   phead->pleft = phead;
   phead->pright = phead;

   /* key value of head is 0 */
   for (ii = 0; ii < KEYLENGTH; ii++) {
      phead->uckey[ii] = 0;
   }

   return phead;
}

/****f* patricia_funcs.c/measurement::patriciasearch
 * SYNOPSIS
 * patriciastruct *patriciasearch(unsigned char *pucsearchkey,
 *                                 patriciastruct *pkeys)
 * DESCRIPTION
 * The function searches the key named "pucsearchkey" in the 
 * patricia trie "pkeys."
 ***/
patriciastruct *patriciasearch(unsigned char *pucsearchkey,
                               patriciastruct * pkeys) {
   /* variables for navigating in the trie */
   patriciastruct *pprekey, *pnowkey;

   IDL(1, printf("reached function patriciasearch\n"));

   /* pointer keys should not be changed */
   pnowkey = pkeys;

   /* going down the trie until... */
   do {
      /* saving key to reconstruct if the next step is made up- or downward */
      pprekey = pnowkey;
      /* going to the left or the right depending on the bit of the trie level
       * in the searched key */
      if ((pucsearchkey[pnowkey->ibit / 8] >> pnowkey->ibit % 8) % 2) {
         pnowkey = pnowkey->pright;
      } else {
         pnowkey = pnowkey->pleft;
      }
   } while (pprekey->ibit > pnowkey->ibit);     /* ... a step upward was made */

   IDL(1, printf("completed function patriciasearch\n"));

   /* this is the only element that COULD fit with the searched one */
   return pnowkey;
}

/****f* patricia_funcs.c/measurement::patriciainsert
 * SYNOPSIS
 * int patriciainsert(unsigned char *pucinskey, patriciastruct *pkeys)
 * DESCRIPTION
 * The function inserts "pucinskey" on a correct place in the
 *  patricia-trie "pkeys".
 ***/
int patriciainsert(unsigned char *pucinskey, patriciastruct * pkeys) {
   /* variables for navigating in the trie and for creating the new node */
   patriciastruct *pprekey, *phelpkey, *pnowkey;
   /* variable for loop and buffering */
   int ii;

   IDL(1, printf("reached function patriciainsert\n"));

   /* pointer keys should not be changed */
   pnowkey = pkeys;

   /* searching key that we want to insert */
   phelpkey = patriciasearch(pucinskey, pnowkey);
   /* if it already exists ... */
   if (patriciacompare(pucinskey, phelpkey->uckey)) {
      IDL(1, printf("completed function patriciainsert\n"));
      /* ... no use inserting it again */
      return 0;
   } else {
      /* if it does not exist -> the key that is "most similar" has been found */
      ii = KEYLENGTH * 8 - 1;
      /* finding first (bit-)position where the 2 keys differ */
      while ((pucinskey[ii / 8] >> ii % 8) % 2 ==
             (phelpkey->uckey[ii / 8] >> ii % 8) % 2) {
         ii--;
      }
      /* going downward in the trie to the position defined by the differing
       * bit or to the lowest level of the trie */
      do {
         pprekey = pnowkey;
         if ((pucinskey[pnowkey->ibit / 8] >> pnowkey->ibit % 8) % 2 == 0) {
            pnowkey = pnowkey->pleft;
         } else {
            pnowkey = pnowkey->pright;
         }
      } while (pnowkey->ibit > ii && pprekey->ibit > pnowkey->ibit);

      /* creating new node */
      phelpkey = calloc(1, sizeof(patriciastruct));

      /* deciding bit position that makes the key in the node different from
       * all others */
      /* (when looking to all bits until this position) */
      phelpkey->ibit = ii;
      /* copying key to the node */
      for (ii = 0; ii < KEYLENGTH; ii++) {
         phelpkey->uckey[ii] = pucinskey[ii];
      }

      /* inserting node */
      /* setting links pointing from the new node to the trie */
      if ((pucinskey[phelpkey->ibit / 8] >> phelpkey->ibit % 8) % 2 == 0) {
         phelpkey->pleft = phelpkey;
         phelpkey->pright = pnowkey;
      } else {
         phelpkey->pright = phelpkey;
         phelpkey->pleft = pnowkey;
      }

      /* setting link from the node inside the trie to the inserted key */
      if ((pucinskey[pprekey->ibit / 8] >> pprekey->ibit % 8) % 2 == 0) {
         pprekey->pleft = phelpkey;
      } else {
         pprekey->pright = phelpkey;
      }
   }
   IDL(1, printf("completed function patriciainsert\n"));

   /* ready */
   return 0;
}

/****f* patricia_funcs.c/measurement::patriciafree
 * SYNOPSIS
 * void patriciafree(patriciastruct *pdelcandidate)
 * DESCRIPTION
 * The function recursively frees all nodes in a patricia trie
 * that are childs of "delcandidate".
 ***/
void patriciafree(patriciastruct * pdelcandidate) {
   /* debugging level 1: mark begin and end of function */
   IDL(1, printf("reached function patriciafree\n"));

   /* if there is a child -> free it */
   if (pdelcandidate->pleft->ibit < pdelcandidate->ibit) {
      patriciafree(pdelcandidate->pleft);
   }
   if (pdelcandidate->pright->ibit < pdelcandidate->ibit) {
      patriciafree(pdelcandidate->pright);
   }
   /* when all childs are freed -> free yourself */
   free(pdelcandidate);

   /* debugging level 1: mark begin and end of function */
   IDL(1, printf("completed function patriciafree\n"));
}

