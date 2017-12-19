/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 *
 * Kernel: 
 * Contact: benchit@zih.tu-dresden.de
 *******************************************************************/
 
#ifndef DATA_STRUCT_h
#define DATA_STRUCT_h

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

#include <stdio.h>
#include <stdlib.h>

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */

typedef struct mydata
{
   //Number of files
   myinttype files_number;
   char      file_unit[6];
   myinttype file_block_size;
   myinttype file_block_number;
   myinttype number_runs;
   myinttype result_view;
   char      path_script[500];
   char      path_temp[500];
   char      num_threads[15];
   char      protocol[5];
   char      resource[100];
   FILE      *CSV;
} mydata_t;


#endif


/********************************************************************
 * Log-History
 * 
 * $Log: data_struct.h,v $
 * 
 *******************************************************************/
