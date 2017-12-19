/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/search/patricia/C/0/0/binarytree/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Execution time of Insertion / Searching in a patricia-trie
 *         PATRICIA: Practical Algorithm To Retrieve Information
 *                   Coded In Alphanumeric
 *******************************************************************/

#include "patricia.h"

#define NUM_FUNC 2

/* Reads the environment variables used by this kernel. 
 * see interface.h for bi_getenv("name", exit_on_error)
 */
void evaluate_environment(mydata_t * pmydata) {
   char *p = 0;

   /* add additional parameters, except BENCHIT_KERNEL_PROBLEMLIST from the
    * parameters file BENCHIT_KERNEL_PROBLEMLIST will be done in bi_getinfo */
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct) {
   char *p = 0;
   mydata_t *penv;

   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);
   /* additional parameters */
   evaluate_environment(penv);

   infostruct->codesequence =
      bi_strdup
      ("Measure time for insert keys into patricia-trie; measure time for search inserted and random generated keys; ");
   infostruct->kerneldescription =
      bi_strdup("Execution time of Insertion / Searching in a patricia-trie");
   infostruct->xaxistext = bi_strdup("number of elements");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = NUM_FUNC;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   infostruct->yaxistexts[0] = bi_strdup("s");
   infostruct->selected_result[0] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[0] = 10;          // logarythmic axis 10^x
   infostruct->legendtexts[0] = bi_strdup("Insertion");

   /* setting up y axis texts and properties */
   infostruct->yaxistexts[1] = bi_strdup("s");
   infostruct->selected_result[1] = SELECT_RESULT_LOWEST;
   infostruct->base_yaxis[1] = 10;          // logarythmic axis 10^x
   infostruct->legendtexts[1] = bi_strdup("Searching");

   /* free all used space */
   if (penv)
      free(penv);
}

/** Implementation of the bi_init of the BenchIT interface.
 *  Here you have the chance to allocate the memory you need.
 *  It is also possible to allocate the memory at the beginning
 *  of every single measurement and to free the memory thereafter.
 *  But always making use of the same memory is faster.
 *  HAVE A LOOK INTO THE HOWTO !
 */
void *bi_init(int problemSizemax) {
   mydata_t *pmydata = NULL;
   myinttype ii = 0, ij = 0;
   /* variable for the number of keys that are needed */
   myinttype lkeynumber = 0;
   /* array for the created keys */
   unsigned char **ppuckeys = NULL;

   pmydata = (mydata_t *) malloc(sizeof(mydata_t));
   if (pmydata == 0) {
      fprintf(stderr, "Allocation of structure mydata_t failed\n");
      fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);

   lkeynumber = (myinttype) bi_get_list_maxelement();
   pmydata->maxsize = lkeynumber;

   /* initialization for random sequence; */
   srand((unsigned)time(NULL));

   /* creating keys */
   ppuckeys = calloc(lkeynumber, sizeof(char *));
   for (ii = 0; ii < lkeynumber; ii++) {
      ppuckeys[ii] = calloc(KEYLENGTH, sizeof(char));
   }

   for (ii = 0; ii < lkeynumber; ii++) {
      for (ij = KEYLENGTH; ij > (KEYLENGTH / 2); ij--) {
         ppuckeys[ii][ij - 1] = rand() % 256;
      }
      /* this guarantees that "all" keys are different from each other */
      for (ij = 0; ij < (KEYLENGTH / 2); ij++) {
         ppuckeys[ii][ij] = (char)((ii >> ij * 8) & 255);
      }
   }

   pmydata->ppuckeys = ppuckeys;

   return (void *)pmydata;
}

/** The central function within each kernel. This function
 *  is called for each measurement step seperately.
 *  @param  mdpv         a pointer to the structure created in bi_init,
 *                       it is the pointer the bi_init returns
 *  @param  problemSize  the actual problemSize
 *  @param  results      a pointer to a field of doubles, the
 *                       size of the field depends on the number
 *                       of functions, there are #functions+1
 *                       doubles
 *  @return 0 if the measurement was sucessfull, something
 *          else in the case of an error
 */
int bi_entry(void *mdpv, int iproblemSize, double *dresults) {
   /* dstart, dend: the start and end time of the measurement */
   /* dtime: the time for a single measurement in seconds */
   double dstart[NUM_FUNC], dend[NUM_FUNC], dtime[NUM_FUNC];
   /* ii, ij is used for loop iterations */
   myinttype ii = 0, ij = 0;
   /* variable for number of actual key numbers */
   myinttype lakn = 0;
   /* pointer to the begin of the patricia trie and to the key that has been
    * found after searching */
   patriciastruct *phead = NULL, *pmykey = NULL;
   unsigned char **ppuckeys, uckey[KEYLENGTH];

   /* cast void* pointer */
   mydata_t *pmydata = (mydata_t *) mdpv;
   ppuckeys = (unsigned char **)pmydata->ppuckeys;

   /* get current problem size from problemlist */
   lakn = (myinttype) bi_get_list_element(iproblemSize);

   /* check wether the pointer to store the results in is valid or not */
   if (dresults == NULL)
      return 1;

   dresults[0] = (double)lakn;

   /* measurement of the insertion time */
   dstart[0] = bi_gettime();
   /* initializing patricia trie */
   phead = patriciainit();
   /* inserting all elements */
   for (ii = 0; ii < lakn; ii++) {
      patriciainsert(ppuckeys[ii], phead);
   }
   dend[0] = bi_gettime();

   /* measurement of the searching time */
   dstart[1] = bi_gettime();
   /* searching key that have been inserted */
   for (ii = 0; ii < lakn; ii++) {
      pmykey = patriciasearch(ppuckeys[ii], phead);
      /* if the keys are not found -> something went wrong */
      if (!(patriciacompare(pmykey->uckey, ppuckeys[ii]))) {
         printf("Inserted key not found!\n");
      }
   }
   /* searching perchancely created keys */
   for (ii = 0; ii < lakn; ii++) {
      for (ij = 0; ij < KEYLENGTH; ij++) {
         uckey[ij] = rand() % 256;
      }
      pmykey = patriciasearch(uckey, phead);
      /* compared found key with the key that has been searched -> this usually 
       * fails */
      patriciacompare(pmykey->uckey, ppuckeys[ii]);
   }
   dend[1] = bi_gettime();

   /* freeing reserved memory */
   patriciafree(phead);

   /* calculate the used time */
   for (ii = 0; ii < NUM_FUNC; ii++) {
      dtime[ii] = dend[ii] - dstart[ii];
      dtime[ii] -= dTimerOverhead;

      /* If the operation was too fast to be measured by the timer function,
       * mark the result as invalid */
      if (dtime[ii] < dTimerGranularity)
         dtime[ii] = INVALID_MEASUREMENT;

      /* store the results in results[1], results[2], ... [1] for the first
       * function, [2] for the second function and so on ... the index 0 always 
       * * * keeps the value for the x axis */
      dresults[ii + 1] =
         (dtime[ii] != INVALID_MEASUREMENT) ? dtime[ii] : INVALID_MEASUREMENT;
   }

   return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void *mdpv) {
   myinttype ii = 0;
   mydata_t *pmydata = (mydata_t *) mdpv;

   if (pmydata) {
      /* free used pointers */
      for (ii = 0; ii < pmydata->maxsize; ii++) {
         if (pmydata->ppuckeys[ii])
            free(pmydata->ppuckeys[ii]);
      }
      if (pmydata->ppuckeys)
         free(pmydata->ppuckeys);

      free(pmydata);
   }

   return;
}

