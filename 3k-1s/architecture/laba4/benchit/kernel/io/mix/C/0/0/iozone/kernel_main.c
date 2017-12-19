/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/mix/C/0/0/iozone/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"
#include "kernel_main.h"
#include "tools/error.h"
#include <unistd.h>

#define MAX_NR_FCT 22
#define MAX_LINE_LENGTH 255

/* from wor.h */
mydata_t* mdp;

void assert_mdp(){
   if (mdp == NULL) mdp = (mydata_t*)malloc(sizeof(mydata_t));
   if (mdp == NULL){
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); 
      fflush(stderr);
      exit(ENOMEM);
   }
}


/* Reads the environment variables used by this kernel. */
void evaluate_environment(){
    int inumentries = 0;
    int ii = 0;
    char * p;
    char unit;
    assert_mdp();
    
    mdp->filesize_min_unit = (char *)malloc(1);
    mdp->filesize_max_unit = (char *)malloc(1);
    mdp->filesize_inc_unit = (char *)malloc(1);
    
    mdp->filename = bi_getenv("BENCHIT_KERNEL_FILENAME", 1);
    mdp->filesize_min = bi_getenv("BENCHIT_KERNEL_FILESIZE_MIN", 1);
    mdp->filesize_max = bi_getenv("BENCHIT_KERNEL_FILESIZE_MAX", 1);
    mdp->filesize_inc = bi_getenv("BENCHIT_KERNEL_FILESIZE_INCREMENT", 1);
    mdp->min = (unsigned int)strtol(mdp->filesize_min, &(mdp->filesize_min_unit), 10);
    mdp->max = (unsigned int)strtol(mdp->filesize_max, &(mdp->filesize_max_unit), 10);
    mdp->inc = (unsigned int)strtol(mdp->filesize_inc, &(mdp->filesize_inc_unit), 10);
    mdp->recordsize = bi_getenv("BENCHIT_KERNEL_RECORDSIZE", 1);
    mdp->testlist = bi_getenv("BENCHIT_KERNEL_TESTLIST", 1);
    mdp->cachelinesize = bi_getenv("BENCHIT_KERNEL_CACHELINE_SIZE", 1);
    mdp->cachesize = bi_getenv("BENCHIT_KERNEL_CACHE_SIZE", 1);
    mdp->options = bi_getenv("BENCHIT_KERNEL_OPTIONS", 1);

//    printf("min=%lu max=%lu inc=%lu\n", mdp->min, mdp->max, mdp->inc);
    
    unit=*(mdp->filesize_min_unit);
    switch (unit){
        case 'k':
             ii=1;
             break;
        case 'm':
             ii=1024;
             break;
        case 'g':
             ii=1048576;
             break;
        default:
             exit(BENVUNKNOWN);
             break;
    }
    mdp->min = mdp->min * ii;
    
    unit=*(mdp->filesize_max_unit);
    switch (unit){
        case 'k':
             ii=1;
             break;
        case 'm':
             ii=1024;
             break;
        case 'g':
             ii=1048576;
             break;
        default:
             exit(BENVUNKNOWN);
             break;
    }
    mdp->max = mdp->max * ii;
    
    unit=*(mdp->filesize_inc_unit);
    switch (unit){
        case 'k':
             ii=1;
             break;
        case 'm':
             ii=1024;
             break;
        case 'g':
             ii=1048576;
             break;
        default:
             exit(BENVUNKNOWN);
             break;
    }
    mdp->inc = mdp->inc * ii;
//    printf("min=%lu max=%lu inc=%lu\n", mdp->min, mdp->max, mdp->inc);
    
    if (0 == strlen(mdp->testlist)){
        fprintf(stderr, "no tests to commit\n"); 
        fflush(stderr);
        exit(BENVEMPTY);
    }
    
    /* find out how many values are given in the list */
    p = mdp->testlist;
    inumentries=1; /* first commata means 1 entry already found */
    while (p){
        p = strstr(p,",");
        if (p){
            p++;
            inumentries++;
        }
    }
    mdp->nr_tests = inumentries;
    /* allocate aray according to number of entries */
    mdp->testarray = (unsigned int *)malloc(sizeof(unsigned int) * inumentries);
    p = mdp->testlist;
    mdp->testarray[0] = strtol(p, (char **)NULL, 10);
    for (ii=1;ii<inumentries; ii++){
        p = strstr(p,",")+1; /* pointer to next number in string */
        mdp->testarray[ii] = strtol(p, (char **)NULL, 10);
    }
    
}



void bi_getinfo(bi_info * infostruct)
{
   int i = 0, j = 0, iswitch=0;
   /* get environment variables for the kernel */
   printf("\nHello You there\n");
   evaluate_environment();
   infostruct->codesequence = bi_strdup("iozone");
   infostruct->xaxistext = bi_strdup("size of file");
   infostruct->num_measurements = (mdp->max - mdp->min+1)/mdp->inc;
   if((mdp->max - mdp->min+1) % mdp->inc != 0)
     infostruct->num_measurements++;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
//   infostruct->base_xaxis = 0;

   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = MAX_NR_FCT;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
