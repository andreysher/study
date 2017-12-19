/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/sort/quicksort/C/OpenMP/0/simple/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "simple.h"

#ifdef _OPENMP
  #include <omp.h>
#endif

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   char *p = 0;
   mydata_t * penv;
   myinttype ii=0;
   
   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   /* parameter list */
   p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
   bi_parselist(p);

   infostruct->codesequence = bi_strdup("partition, qsort(left), qsort(right)");
   #if (_OMPENMP < 200805)
     printf("\n\nError: OpenMP Version is to old (needed 3.0 or higher)!\nExit kernel.\n"); fflush(stdout);
     exit(1);
   #endif
   #ifdef _OPENMP
     infostruct->kerneldescription = bi_strdup("Quicksort (C+OpenMP)");
     infostruct->kernel_execs_omp = 1;
     infostruct->num_threads_per_process = omp_get_max_threads();
   #else
     infostruct->kerneldescription = bi_strdup("Quicksort (C)");
     infostruct->kernel_execs_omp = 0;
     infostruct->num_threads_per_process = 1;
   #endif
   infostruct->xaxistext = bi_strdup("number of elements");
   infostruct->num_measurements = infostruct->listsize;
   infostruct->num_processes = 1;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = 6;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   for (ii=0; ii<infostruct->numfunctions;ii++)
   {
      infostruct->yaxistexts[ii] = bi_strdup("s");
      infostruct->selected_result[ii] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[ii] = 10; //logarythmic axis 10^x
   } 
   infostruct->legendtexts[0] = bi_strdup("integer (clib)");
   infostruct->legendtexts[1] = bi_strdup("float   (clib)");
   infostruct->legendtexts[2] = bi_strdup("double  (clib)");
   infostruct->legendtexts[3] = bi_strdup("integer (self)");
   infostruct->legendtexts[4] = bi_strdup("float   (self)");
   infostruct->legendtexts[5] = bi_strdup("double  (self)");
/*   
   infostruct->legendtexts[0] = bi_strdup("integer (stack)");
   infostruct->legendtexts[1] = bi_strdup("float   (stack)");
   infostruct->legendtexts[2] = bi_strdup("double  (stack)");
*/
   /* free all used space */
   if (penv) free(penv);
}



/** Implementation of the bi_init of the BenchIT interface.
 *  Here you have the chance to allocate the memory you need.
 *  It is also possible to allocate the memory at the beginning
 *  of every single measurement and to free the memory thereafter.
 *  But always making use of the same memory is faster.
 *  HAVE A LOOK INTO THE HOWTO !
 */
void* bi_init(int problemSizemax)
{
   mydata_t * pmydata;
   myinttype ii, maxsize;

   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }

   maxsize = (myinttype) bi_get_list_maxelement();
   pmydata->maxsize = maxsize;

   pmydata->intarray = (myinttype *)malloc(maxsize * sizeof(myinttype));
   pmydata->floatarray = (float *)malloc(maxsize * sizeof(float));
   pmydata->doublearray = (double *)malloc(maxsize * sizeof(double));
   if ((pmydata->intarray == NULL) || (pmydata->floatarray == NULL) || (pmydata->doublearray == NULL))
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }
   
   /*initialization for random sequence;*/
   srand((unsigned) time(NULL));
   //srandomdev();

   /* fill the lists with random values */
   for (ii = 0; ii < maxsize; ii++)
   {
      pmydata->intarray[ii] = (myinttype)rand();
   }

   for (ii = 0; ii < maxsize; ii++)
   {  //((random() + random() / random()) * pow(random(), 3))
      pmydata->floatarray[ii] = (float) (pow(rand(), 2) / rand());
   }

   for (ii = 0; ii < maxsize; ii++)
   {
      pmydata->doublearray[ii] = (double) (pow(rand(), 2) / rand());
   }
   
/*
      printf("\nInteger-Numbers (%d)\n",maxsize);
      for (ii=0; ii<maxsize; ii += 5)
        {
		   printf("%d:\t %d\t %d\t %d\t %d\t %d\n",ii,pmydata->intarray[ii],pmydata->intarray[ii+1],pmydata->intarray[ii+2],pmydata->intarray[ii+3],pmydata->intarray[ii+4]);
		}

      printf("\nFloat-Numbers (%d)\n",maxsize);
      for (ii=0; ii<maxsize; ii += 5)
        {
		   printf("%d:\t %.1f\t\t %.1f\t\t %.1f\t\t %.1f\t\t %.1f\n",ii,pmydata->floatarray[ii],pmydata->floatarray[ii+1],pmydata->floatarray[ii+2],pmydata->floatarray[ii+3],pmydata->floatarray[ii+4]);
		}

      printf("\nDouble-Numbers (%d)\n",maxsize);
      for (ii=0; ii<maxsize; ii += 5)
        {
		   printf("%d:\t %.2f\t\t %.2f\t\t %.2f\t\t %.2f\t\t %.2f\n",ii,pmydata->doublearray[ii],pmydata->doublearray[ii+1],pmydata->doublearray[ii+2],pmydata->doublearray[ii+3],pmydata->doublearray[ii+4]);
		}

*/

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
int bi_entry(void * mdpv, int iproblemSize, double * dresults)
{
  /* dstart, dend: the start and end time of the measurement */
  /* dtime: the time for a single measurement in seconds */
  double dstart = 0.0, dend = 0.0, dtime = 0.0;
  /* flops stores the calculated FLOPS */
  double dres = 0.0;
  /* ii is used for loop iterations */
  myinttype ii = 0, imyproblemSize = 0;
  /* cast void* pointer */
  mydata_t * pmydata = (mydata_t *) mdpv;
  /* lists */
  myinttype * intarray;
  float * floatarray;
  double * doublearray;

  /* calculate real problemSize */
  imyproblemSize = (myinttype) bi_get_list_element(iproblemSize);

  /* check wether the pointer to store the results in is valid or not */
  if (dresults == NULL) return 1;

  /* create temp-array's which will be filled and then sorted */
  intarray = (myinttype *) malloc(sizeof(myinttype) * imyproblemSize);
  floatarray = (float *) malloc(sizeof(float) * imyproblemSize);
  doublearray =  (double *) malloc(sizeof(double) * imyproblemSize);
  if ((intarray == NULL) || (floatarray == NULL) || (doublearray == NULL))
  {
     fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
     exit(127);
  }

  for (ii=0; ii<imyproblemSize; ii++)
  {
    intarray[ii] = pmydata->intarray[ii];
	floatarray[ii] = pmydata->floatarray[ii];
	doublearray[ii] = pmydata->doublearray[ii];
  }
  /* check wether values are unsorted*/
  ii = 0;
  ii += verify_int(intarray, imyproblemSize);
  ii += verify_float(floatarray, imyproblemSize);
  ii += verify_double(doublearray, imyproblemSize);
//  if (ii != 3) fprintf(stderr, "\n -- values are unsorted (clib-qsort) --\n");
  
  dresults[0] = (double)imyproblemSize;

  dstart = bi_gettime(); 
  qsort(intarray, imyproblemSize, sizeof(myinttype), quicksort_clib_myinttype);
  dend = bi_gettime();
  dtime = dend - dstart;
  dtime -= dTimerOverhead;
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;
  dresults[1] = dtime;

  dstart = bi_gettime(); 
  qsort(floatarray, imyproblemSize, sizeof(float), quicksort_clib_flt);
  dend = bi_gettime();
  dtime = dend - dstart;
  dtime -= dTimerOverhead;
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;
  dresults[2] = dtime;

  dstart = bi_gettime(); 
  qsort(doublearray, imyproblemSize, sizeof(double), quicksort_clib_dbl);
  dend = bi_gettime();
  dtime = dend - dstart;
  dtime -= dTimerOverhead;
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;
  dresults[3] = dtime;
  
  /* check values */
  ii = 0;
  ii += verify_int(intarray, imyproblemSize);
  ii += verify_float(floatarray, imyproblemSize);
  ii += verify_double(doublearray, imyproblemSize);
//  if (ii = 3) fprintf(stderr, "\n -- values are sorted (clib-qsort) --\n");
  if (ii != 3) fprintf(stderr, "\nverification of sorted lists (clib-qsort) failed!!!\n");

  /* re-init unsorted values */ 
  for (ii=0; ii<imyproblemSize; ii++)
  {
    intarray[ii] = pmydata->intarray[ii];
	floatarray[ii] = pmydata->floatarray[ii];
	doublearray[ii] = pmydata->doublearray[ii];
  }
  /* check wether values are unsorted*/
  ii = 0;
  ii += verify_int(intarray, imyproblemSize);
  ii += verify_float(floatarray, imyproblemSize);
  ii += verify_double(doublearray, imyproblemSize);
//  if (ii != 3) fprintf(stderr, "\n -- values are unsorted (self-qsort) --\n");

  dstart = bi_gettime(); 
//  quicksort_int(intarray, 0, imyproblemSize);
  quicksort_wikipedia_int_parallel(intarray, 0, imyproblemSize-1);
  dend = bi_gettime();
//printf("\nIntarray finished:\n");
  dtime = dend - dstart;
  dtime -= dTimerOverhead;
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;
  dresults[4] = dtime;

  dstart = bi_gettime(); 
//  quicksort_float(floatarray, 0, imyproblemSize);
  quicksort_wikipedia_flt_parallel(floatarray, 0, imyproblemSize-1);
  dend = bi_gettime();
//printf("\nFloatarray finished:\n");
  dtime = dend - dstart;
  dtime -= dTimerOverhead;
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;
  dresults[5] = dtime;

  dstart = bi_gettime(); 
//  quicksort_double(doublearray, 0, imyproblemSize);
  quicksort_wikipedia_dbl_parallel(doublearray, 0, imyproblemSize-1);
  dend = bi_gettime();
//printf("\nDoublearray finished:\n");
  dtime = dend - dstart;
  dtime -= dTimerOverhead;
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;
  dresults[6] = dtime;

//for (ii=0; ii<imyproblemSize; ii++) printf("%d %d\n",intarray[ii],pmydata->intarray[ii]);

  /* check values */
  if (!(verify_int(intarray, imyproblemSize))) 
  {
	  printf("\nverification of sorted lists (selfmade integer quicksort) failed!!!\n");
	  printf("\nIntarray:\n");
      for (ii=0; ii<imyproblemSize; ii += 5)
        {
		   printf("%d %d %d %d %d\n",intarray[ii],intarray[ii+1],intarray[ii+2],intarray[ii+3],intarray[ii+4]);
//		   printf("%d %d %d %d %d\n\n",pmydata->intarray[ii],pmydata->intarray[ii+1],pmydata->intarray[ii+2],pmydata->intarray[ii+3],pmydata->intarray[ii+4]);
		}
  }
  if (!(verify_float(floatarray, imyproblemSize))) 
  {
	  printf("\nverification of sorted lists (selfmade float quicksort) failed!!!\n");
	  printf("\nFloatarray:\n");
      for (ii=0; ii<imyproblemSize; ii += 5)
        {
		   printf("%f %f %f %f %f\n",floatarray[ii],floatarray[ii+1],floatarray[ii+2],floatarray[ii+3],floatarray[ii+4]);
		}
  }
  if (!(verify_double(doublearray, imyproblemSize))) 
  {
	  printf("\nverification of sorted lists (selfmade double quicksort) failed!!!\n");
	  printf("\ndoublearray:\n");
      for (ii=0; ii<imyproblemSize; ii++)
        {
		   printf("%f\t\t%f\n",doublearray[ii],pmydata->doublearray[ii]);
		}
  }
  	
  if(intarray) free(intarray);
  if(doublearray) free(doublearray);
  if(floatarray) free(floatarray);
  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   mydata_t * pmydata = (mydata_t*)mdpv;
   if (pmydata->intarray) free(pmydata->intarray);
   if (pmydata->floatarray) free(pmydata->floatarray);
   if (pmydata->doublearray) free(pmydata->doublearray);
   if (pmydata) free(pmydata);
   return;
}
