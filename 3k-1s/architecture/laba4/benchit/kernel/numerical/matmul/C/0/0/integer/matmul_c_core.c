/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: matmul_c_core.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/C/0/0/integer/matmul_c_core.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: blocked Matrix Multiplication (C)
 *******************************************************************/

#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "matmul.h"
#include "interface.h"

void multaijk_(int *a, int *b, int *c, int *size);
void multaikj_(int *a, int *b, int *c, int *size);
void multajik_(int *a, int *b, int *c, int *size);
void multajki_(int *a, int *b, int *c, int *size);
void multakji_(int *a, int *b, int *c, int *size);
void multakij_(int *a, int *b, int *c, int *size);

double getlanguage_(void);

void multaijk_(int *a, int *b, int *c, int *size)
{
  int i, j, k;
  int s = *size;
  for (i = 0; i < s; i++)
    for (j = 0; j < s; j++)
      for (k = 0; k < s; k++)
      {
        c[ i * s + j ] = c[ i * s + j ] + a[ i * s + k ] * b[ k * s + j ];
      }
}

void multaikj_(int *a, int *b, int *c, int *size)
{
  int i, j, k;
  int s = *size;
  for (i = 0; i < s; i++)
    for (k = 0; k < s; k++)
      for (j = 0; j < s; j++)
      {
        c[ i * s + j ] = c[ i * s + j ] + a[ i * s + k ] * b[ k * s + j ];
      }
}

void multajik_(int *a, int *b, int *c, int *size)
{
  int i, j, k;
  int s = *size;
  for (j = 0; j < s; j++)
    for (i = 0; i < s; i++)
      for (k = 0; k < s; k++)
      {
        c[ i * s + j ] = c[ i * s + j ] + a[ i * s + k ] * b[ k * s + j ];
      }
}

void multajki_(int *a, int *b, int *c, int *size)
{
  int i, j, k;
  int s = *size;
  for (j = 0; j < s; j++)
    for (k = 0; k < s; k++)
      for (i = 0; i < s; i++)
      {
        c[ i * s + j ] = c[ i * s + j ] + a[ i * s + k ] * b[ k * s + j ];
      }
}

void multakij_(int *a, int *b, int *c, int *size)
{
  int i, j, k;
  int s = *size;
  for (k = 0; k < s; k++)
    for (i = 0; i < s; i++)
      for (j = 0; j < s; j++)
      {
        c[ i * s + j ] = c[ i * s + j ] + a[ i * s + k ] * b[ k * s + j ];
      }
}

void multakji_(int *a, int *b, int *c, int *size)
{
  int i, j, k;
  int s = *size;
  for (k = 0; k < s; k++)
    for (j = 0; j < s; j++)
      for (i = 0; i < s; i++)
      {
        c[ i * s + j ] = c[ i * s + j ] + a[ i * s + k ] * b[ k * s + j ];
      }
}

double getlanguage_()
{
  return 1.0;
}

