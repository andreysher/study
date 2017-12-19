/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matmul_sub.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/0/0/integer/matmul_sub.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: blocked Matrix Multiplication (C)
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "interface.h"
#include "matmul.h"


/** These variables will help us to keep the overview over the arrays
  * we access for our functions/data.
  */
/* Number of different ways an algorithm will be measured.
   Example: loop orders: ijk, ikj, jki, jik, kij, kji -> functionCount=6 with
   each different loop order in an own function. */
int functionCount;
/* Number of fixed functions we have per measurement.
   Example: execution time and MFLOPS are measured for each loop order
   -> valuesPerFunction=2 */
int valuesPerFunction;


typedef struct floating_data_struct
{
  myinttype *feld1, *feld2, *feld3;
}
fds;

void (*entry1) (myinttype *a, myinttype *b, myinttype *c, int *size);
extern void multaijk_(myinttype *a, myinttype *b, myinttype *c, int *size);
extern void multaikj_(myinttype *a, myinttype *b, myinttype *c, int *size);
extern void multajik_(myinttype *a, myinttype *b, myinttype *c, int *size);
extern void multajki_(myinttype *a, myinttype *b, myinttype *c, int *size);
extern void multakij_(myinttype *a, myinttype *b, myinttype *c, int *size);
extern void multakji_(myinttype *a, myinttype *b, myinttype *c, int *size);
extern double getlanguage_(void);

void init_(fds *myfds, int *size);
int getnumberofversions_(void);
void useversion_(int *version);
void entry_(void *ptr, int *size);
double count_(int *version, int *size);
int coreerror(char *string);
double getseqentryoverhead(void *mem);

void init_(fds *myfds, int *size)
{
  register int x, y;
  long index;

  IDL(3, printf("field size: %ld bytes", (long) * size * (*size) * sizeof(int)));
  for (x = 0; x < *size; x++)
    for (y = 0; y < *size; y++)
    {
      index = x * (*size) + y;
      IDL(5, printf("%ld\n", index));
      /* Feld voller Zahlen zwischen 0 und 9 */
      myfds->feld1[ index ] = x%10;
      /* Ein Feld voller Zahlen zwischen -3 und 3 */
      myfds->feld2[ index ] = (y%7)-3;
      myfds->feld3[ index ] = 0;
    }
  IDL(3, printf("init fertig\n"));
}

int getnumberofversions_()
{
  return functionCount;
}

void useversion_(int *version)
{
  switch (*version)
  {
    case 0:
      entry1 = multaijk_;
      break;
    case 1:
      entry1 = multaikj_;
      break;
    case 2:
      entry1 = multajik_;
      break;
    case 3:
      entry1 = multajki_;
      break;
    case 4:
      entry1 = multakij_;
      break;
    case 5:
      entry1 = multakji_;
      break;
  }
}

void entry_(void *ptr, int *size)
{
  fds * myfds = ptr;
  myinttype *f1 = myfds->feld1, *f2 = myfds->feld2, *f3 = myfds->feld3;
  if (*size == 0)
    return ;
  else
    entry1(f1, f2, f3, size);
}

double count_(int *version, int *size)
{
  double ulSize = 1.0 * *size;
  switch (*version)
  {
    default:
      return 2.0 * (ulSize) * (ulSize) * (ulSize);
  }
}

void bi_getinfo(bi_info* infostruct)
{
  char *p = 0;
  int i, j;

  /* get environment variables for the kernel */
  /* parameter list */
  p = bi_getenv("BENCHIT_KERNEL_PROBLEMLIST", 0);
  bi_parselist(p);

  infostruct->codesequence = bi_strdup("for(i=0; i<s; i++)#"
                                        "  for(j=0; j<s; j++)#"
                                        "    for(k=0; k<s; k++)#"
                                        "    {#"
                                        "      c[j*s+i]+=a[k*s+i]*b[j*s+k];#"
                                        "    }");
  infostruct->xaxistext = bi_strdup("Matrix Size");
  infostruct->kerneldescription = bi_strdup("Matrix Multiply (C) int");
  infostruct->num_measurements = infostruct->listsize;
  infostruct->numfunctions = 6;
  infostruct->kernel_execs_mpi1 = 0;
  infostruct->kernel_execs_mpi2 = 0;
  infostruct->kernel_execs_pvm = 0;
  infostruct->kernel_execs_omp = 0;
  infostruct->kernel_execs_pthreads = 0;
  /* B ########################################################*/
  functionCount = 6;
  valuesPerFunction = 1;
  /*########################################################*/
  infostruct->numfunctions = functionCount * valuesPerFunction;

  /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
  for (j = 0; j < functionCount; j++)
  {
    /* B ########################################################*/
    int index1 = 0 * functionCount + j;
    /* 1st function */
    infostruct->yaxistexts[ index1 ] = bi_strdup("IOPS");
    infostruct->selected_result[index1] = SELECT_RESULT_HIGHEST;
    infostruct->base_yaxis[ index1 ] = 0;
    /*########################################################*/
    switch (j)
    {
        /* B ########################################################*/
      case 0:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("IOPS (ijk)");
        break;
      case 1:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("IOPS (ikj)");
        break;
      case 2:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("IOPS (jik)");
        break;
      case 3:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("IOPS (jki)");
        break;
      case 4:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("IOPS (kij)");
        break;
      case 5:
        infostruct->legendtexts[ index1 ] =
          bi_strdup("IOPS (kji)");
        break;
      default:
        fprintf(stderr, "Should not reach default section of case.\n");
        fflush(stderr);
        exit(127);
        /*########################################################*/
    }
  }
  if (DEBUGLEVEL > 3)
  {
    /* the next for loop: */
    /* this is for your information only and can be ereased if the kernel works fine */
    for (i = 0; i < infostruct->numfunctions; i++)
    {
      printf("yaxis[%2d]=%s\t\t\tlegend[%2d]=%s\n",
              i, infostruct->yaxistexts[ i ], i, infostruct->legendtexts[ i ]);
    }
  }

}

double getseqentryoverhead(void *mem)
{
  double start, stop, diff;
  int nu = 0, s;

  init_(mem, &nu);
  start = bi_gettime();
  for (s = 0; s < 1000; s++)
  {
    entry_(mem, &nu);
  }
  stop = bi_gettime();
  diff = stop - start - dTimerOverhead;
  if(diff < dTimerGranularity)
    diff = 0.0;
  return diff / 1000;
}

int bi_entry(void* mdpv, int problemSize, double* results)
{
  static double calloverhead = 0;

  int v,w,numberOfRuns = 1;
  double time = 0;
  unsigned long count = 0;
  double start, stop;

  /* calculate real problemSize */
  problemSize = (int)bi_get_list_element(problemSize);

  /* check wether the pointer to store the results in is valid or not */
  if (results == NULL)
    return 1;

  results[ 0 ] = problemSize;

  count = count_(&v, &problemSize);
  for (v = 0; v < functionCount; v++)
  {
    useversion_(&v);
    calloverhead = getseqentryoverhead(mdpv);
    do
    {
      init_(mdpv, &problemSize);
      start = bi_gettime();
      for (w=0;w<numberOfRuns;w++)
    	  entry_(mdpv, &problemSize);
      stop = bi_gettime();
      time = stop - start;
      time -= dTimerOverhead;
      time -= numberOfRuns*calloverhead;
      numberOfRuns=numberOfRuns*4;
  
    } while ((time<dTimerGranularity)&&(numberOfRuns<256));
      numberOfRuns=numberOfRuns/4;
    /* If the measured time is smaller than the resolution of our timer,
     * mark the result as invalid
     */
    if (time < dTimerGranularity)
      results[ v + 1 ] = INVALID_MEASUREMENT;
    else
      results[ v + 1 ] = ((double) count) / time;
  }
  return 0;
}

int coreerror(char *string)
{
  printf("Core Error: %s\n", string);
  return 1;
}

void *bi_init(int problemSizemax)
{

  fds * myfds;

  IDL(2, printf("enter bi_init\n"));
  myfds = (fds*) malloc(sizeof(fds));
  IDL(3, printf("allocating structure myfds\n"));
  if (myfds == NULL)
  {
    printf("allocation of structure myfds failed\n");
    exit(127);
  }

  /* calculate real maximum problem size
     problemSizemax might be smaller than BENCHIT_KERNEL_PROBLEMSIZE_MAX
     if BENCHIT_KERNEL_PROBLEMSIZE_INCREMENT is greater than one */

  problemSizemax = (int)bi_get_list_maxelement();

  myfds->feld1 = (int*) malloc(problemSizemax*problemSizemax * sizeof(int));
  IDL(3, printf("Alloc 1 done\n"));
  myfds->feld2 = (int*) malloc(problemSizemax*problemSizemax * sizeof(int));
  IDL(3, printf("Alloc 2 done\n"));
  myfds->feld3 = (int*) malloc(problemSizemax*problemSizemax * sizeof(int));
  IDL(3, printf("Alloc 3 done\n"));

  if ((myfds->feld1 == NULL) || (myfds->feld2 == NULL) ||
       (myfds->feld3 == NULL))
  {
    printf("malloc (%ld bytes) failed in bi_init()\n",
            (long) (3.0 * problemSizemax * problemSizemax * sizeof(int)));
    bi_cleanup(myfds);
    exit(127);
  }

  IDL(2, printf("leave bi_init\n"));
  return (myfds);
}

void bi_cleanup(void* mdpv)
{
  fds * data = mdpv;
  IDL(3, printf("cleaning..."))
  if (data != NULL)
  {
    IDL(3, printf("1"))
    if (data->feld1 != NULL)
    {
      free(data->feld1);
      data->feld1 = NULL;
    }
    IDL(3, printf("2"))
    if (data->feld2 != NULL)
    {
      free(data->feld2);
      data->feld2 = NULL;
    }
    IDL(3, printf("3"))
    if (data->feld3 != NULL)
    {
      free(data->feld3);
      data->feld3 = NULL;
    }
    IDL(3, printf("4\n"))
    free(data);
  }
}


