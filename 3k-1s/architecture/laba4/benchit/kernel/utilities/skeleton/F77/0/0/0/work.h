/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: work.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/utilities/skeleton/C/0/0/0/work.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Fortran kernel skeleton
 *******************************************************************/

#ifndef __work_h
#define __work_h

#define myinttype int

extern void work_1_(myinttype*);
extern void work_2_(myinttype*);

/** The data structure that holds all the data.
 *  Please use this construct instead of global variables.
 *  Global Variables seem to have large access times (?).
 */
typedef struct mydata
{
   myinttype dummy;
/*
   <variabletype1> <variablename1>;
   <variabletype2> <variablename2>;
*/
} mydata_t;

#endif
