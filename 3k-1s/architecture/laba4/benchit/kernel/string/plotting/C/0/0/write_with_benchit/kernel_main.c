/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/string/plotting/C/0/0/write_with_benchit/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: writing with BenchIT
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"

typedef struct letter{
   char character;
   int numPoints;
   int *points;
} letter_t;

typedef struct mydata{
   int num;
   int newlines;
   int globalPoints;
   char *str;
   letter_t * myletters;
} mydata_t;

int allLetter = 28;
//int allLetter = 47;
int letterFirstPosX = 1, letterFirstPosY = 1;
int letterNextPos = 0;
int nextLetter = 0;

void initLetter(letter_t *L, char *c);
void getPoint(double *outX, double *outY, int *x0, int *y0, int *nextPointOfLetter, int *nextLetter, letter_t *L);

/* Reads the environment variables used by this kernel. */
void evaluate_environment(mydata_t * pmydata)
{
   int i, j, sum=0, errors = 0;
   char * p = 0;
   char c[1];
   
   pmydata->myletters = (letter_t*)malloc(allLetter * sizeof(letter_t));

#ifdef haha
   c[0] = (char)65;  /* A - Z */
   for(i=0; i<26; i++){
     initLetter(&(pmydata->myletters[i]), c);
     c[0] = (char)((int)c[0]+1);
   }
   
   c[0] = (char)48;  /* 0 - 9 */
   for(i=26; i<36; i++){
     initLetter(&(pmydata->myletters[i]), c);
     c[0] = (char)((int)c[0]+1);
   }

   i = 36;
   initLetter(&(pmydata->myletters[i]), " "); i++;  /*   */
   initLetter(&(pmydata->myletters[i]), "."); i++;  /* . */
   initLetter(&(pmydata->myletters[i]), "!"); i++;  /* ! */
   initLetter(&(pmydata->myletters[i]), "?"); i++;  /* ? */
   initLetter(&(pmydata->myletters[i]), "*"); i++;  /* * */
   initLetter(&(pmydata->myletters[i]), "+"); i++;  /* + */
   initLetter(&(pmydata->myletters[i]), "-"); i++;  /* - */
   initLetter(&(pmydata->myletters[i]), "="); i++;  /* = */
   initLetter(&(pmydata->myletters[i]), "/"); i++;  /* / */
   initLetter(&(pmydata->myletters[i]), "\\"); i++; /* \ */
   initLetter(&(pmydata->myletters[i]), "^"); i++;  /* ^ */
#endif 
  
   i = 0; 
   initLetter(&(pmydata->myletters[i]), "A"); i++;  /* A */
   initLetter(&(pmydata->myletters[i]), "B"); i++;  /* B */
   initLetter(&(pmydata->myletters[i]), "C"); i++;  /* C */
   initLetter(&(pmydata->myletters[i]), "D"); i++;  /* D */
   initLetter(&(pmydata->myletters[i]), "E"); i++;  /* E */
   initLetter(&(pmydata->myletters[i]), "F"); i++;  /* F */
   initLetter(&(pmydata->myletters[i]), "G"); i++;  /* G */
   initLetter(&(pmydata->myletters[i]), "H"); i++;  /* H */
   initLetter(&(pmydata->myletters[i]), "I"); i++;  /* I */
   initLetter(&(pmydata->myletters[i]), "J"); i++;  /* J */
   initLetter(&(pmydata->myletters[i]), "K"); i++;  /* K */
   initLetter(&(pmydata->myletters[i]), "L"); i++;  /* L */
   initLetter(&(pmydata->myletters[i]), "M"); i++;  /* M */
   initLetter(&(pmydata->myletters[i]), "N"); i++;  /* N */
   initLetter(&(pmydata->myletters[i]), "O"); i++;  /* O */
   initLetter(&(pmydata->myletters[i]), "P"); i++;  /* P */
   initLetter(&(pmydata->myletters[i]), "Q"); i++;  /* Q */
   initLetter(&(pmydata->myletters[i]), "R"); i++;  /* R */
   initLetter(&(pmydata->myletters[i]), "S"); i++;  /* S */
   initLetter(&(pmydata->myletters[i]), "T"); i++;  /* T */
   initLetter(&(pmydata->myletters[i]), "U"); i++;  /* U */
   initLetter(&(pmydata->myletters[i]), "V"); i++;  /* V */
   initLetter(&(pmydata->myletters[i]), "W"); i++;  /* W */
   initLetter(&(pmydata->myletters[i]), "X"); i++;  /* X */
   initLetter(&(pmydata->myletters[i]), "Y"); i++;  /* Y */
   initLetter(&(pmydata->myletters[i]), "Z"); i++;  /* Z */

   initLetter(&(pmydata->myletters[i]), " "); i++;  /*   */
   initLetter(&(pmydata->myletters[i]), "!"); i++;  /* ! */
#ifdef zuviel
   initLetter(&(pmydata->myletters[i]), "0"); i++;  /* 0 */
   initLetter(&(pmydata->myletters[i]), "1"); i++;  /* 1 */
   initLetter(&(pmydata->myletters[i]), "2"); i++;  /* 2 */
   initLetter(&(pmydata->myletters[i]), "3"); i++;  /* 3 */
   initLetter(&(pmydata->myletters[i]), "4"); i++;  /* 4 */
   initLetter(&(pmydata->myletters[i]), "5"); i++;  /* 5 */
   initLetter(&(pmydata->myletters[i]), "6"); i++;  /* 6 */
   initLetter(&(pmydata->myletters[i]), "7"); i++;  /* 7 */
   initLetter(&(pmydata->myletters[i]), "8"); i++;  /* 8 */
   initLetter(&(pmydata->myletters[i]), "9"); i++;  /* 9 */

   initLetter(&(pmydata->myletters[i]), " "); i++;  /*   */
   initLetter(&(pmydata->myletters[i]), "."); i++;  /* . */
   initLetter(&(pmydata->myletters[i]), "!"); i++;  /* ! */
   initLetter(&(pmydata->myletters[i]), "?"); i++;  /* ? */
   initLetter(&(pmydata->myletters[i]), "*"); i++;  /* * */
   initLetter(&(pmydata->myletters[i]), "+"); i++;  /* + */
   initLetter(&(pmydata->myletters[i]), "-"); i++;  /* - */
   initLetter(&(pmydata->myletters[i]), "="); i++;  /* = */
   initLetter(&(pmydata->myletters[i]), "/"); i++;  /* / */
   initLetter(&(pmydata->myletters[i]), "\\"); i++; /* \ */
   initLetter(&(pmydata->myletters[i]), "^"); i++;  /* ^ */
#endif

   p = bi_getenv("NUMBER_LETTERS", 0);
   if(p == NULL) errors++;
   else pmydata->num = atoi(p);
   
   p = bi_getenv("PLOTTING_STRING", 0);
   if(p == NULL) errors++;
   else{
     pmydata->newlines = 0;
     for(i=0; i<pmydata->num; i++){
       if((int)p[i]==59){           /* end-of-string character is ";" */
         pmydata->num = i;
         break;
       } else if((int)p[i]==95){    /* newline character is "_" */
         pmydata->newlines++;
       }
     }
     if(i==pmydata->num-1){
       fprintf(stderr, "There's no end-of-string character in the string, or the value NUMBER_LETTERS is to small!\n");
       exit(1);
     }
     pmydata->str = (char*)malloc(pmydata->num * sizeof(char));
     for(i=0; i<pmydata->num; i++){
       pmydata->str[i] = p[i];
       for(j=0; j<allLetter; j++){
         if(p[i]==pmydata->myletters[j].character) sum += pmydata->myletters[j].numPoints;
       }
     }
   }

   if(errors > 0){
      fprintf(stderr, "There's at least one environment variable not set!\n");
      exit(1);
   }
   
   pmydata->globalPoints = sum;
}

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   int i;
   mydata_t * penv;
   
   penv = (mydata_t *) malloc(sizeof(mydata_t));

   /* get environment variables for the kernel */
   evaluate_environment(penv);
   infostruct->codesequence = bi_strdup("start kernel; write letters; show letters in results window ");
   infostruct->kerneldescription = bi_strdup("writing with BenchIT");
   infostruct->xaxistext = bi_strdup("");
   infostruct->num_measurements = penv->globalPoints;
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = 0;
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 0;
   infostruct->numfunctions = allLetter;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
   for(i=0; i<allLetter; i++) infostruct->yaxistexts[i] = bi_strdup(""); infostruct->legendtexts[i] = bi_strdup("");

   /* free all used space */
   if (penv) bi_cleanup((void*)penv);
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
   
   pmydata = (mydata_t*)malloc(sizeof(mydata_t));
   if (pmydata == 0)
   {
      fprintf(stderr, "Allocation of structure mydata_t failed\n"); fflush(stderr);
      exit(127);
   }

   evaluate_environment(pmydata);
   letterFirstPosY += pmydata->newlines * 10;

   return (void *)pmydata;
}

/* Fieldcoding:
      _ _ _ _ _
     |_|_|_|_|_| +6
     |_|_|_|_|_|
     |_|_|_|_|_|  :
     |_|_|_|_|_|  :
     |_|_|_|_|_|
     |_|_|_|_|_| +1
     |_|_|_|_|_| +0
      0 7 ... 28
*/
void initLetter(letter_t *L, char *c){
   int *p;
   
   switch((int)(c[0])){
     /* Buchstaben */
     case 65: L->character = c[0];
              L->numPoints = 10+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=2; p[2]=11; p[3]=20; p[4]=25; p[5]=30; p[6]=2; p[7]=16; p[8]=30; p[9]=28; break;
     case 66: L->character = c[0];
              L->numPoints = 13+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=3; p[2]=6; p[3]=20; p[4]=33; p[5]=32; p[6]=17; p[7]=3; p[8]=17; p[9]=30; p[10]=29; p[11]=14; p[12]=0; break;
     case 67: L->character = c[0];
              L->numPoints = 8+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=29; p[1]=21; p[2]=7; p[3]=2; p[4]=4; p[5]=13; p[6]=27; p[7]=33; break;
     case 68: L->character = c[0];
              L->numPoints = 8+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=3; p[2]=6; p[3]=20; p[4]=32; p[5]=31; p[6]=14; p[7]=0; break;
     case 69: L->character = c[0];
              L->numPoints = 10+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=28; p[1]=14; p[2]=0; p[3]=3; p[4]=17; p[5]=31; p[6]=3; p[7]=6; p[8]=20; p[9]=34; break;
     case 70: L->character = c[0];
              L->numPoints = 7+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=3; p[2]=24; p[3]=3; p[4]=6; p[5]=20; p[6]=34; break;
     case 71: L->character = c[0];
              L->numPoints = 10+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=17; p[1]=31; p[2]=30; p[3]=21; p[4]=7; p[5]=2; p[6]=4; p[7]=13; p[8]=27; p[9]=33; break;
     case 72: L->character = c[0];
              L->numPoints = 7+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=6; p[2]=3; p[3]=17; p[4]=31; p[5]=34; p[6]=28; break;
     case 73: L->character = c[0];
              L->numPoints = 7+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=13; p[1]=27; p[2]=20; p[3]=17; p[4]=14; p[5]=7; p[6]=21; break;
     case 74: L->character = c[0];
              L->numPoints = 6+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=13; p[1]=34; p[2]=30; p[3]=21; p[4]=7; p[5]=2; break;
     case 75: L->character = c[0];
              L->numPoints = 9+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=6; p[2]=3; p[3]=18; p[4]=34; p[5]=18; p[6]=3; p[7]=16; p[8]=28; break;
     case 76: L->character = c[0];
              L->numPoints = 5+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=6; p[1]=3; p[2]=0; p[3]=14; p[4]=28; break;
     case 77: L->character = c[0];
              L->numPoints = 7+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=3; p[2]=6; p[3]=17; p[4]=34; p[5]=31; p[6]=28; break;
     case 78: L->character = c[0];
              L->numPoints = 7+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=3; p[2]=6; p[3]=17; p[4]=28; p[5]=31; p[6]=34; break;
     case 79: L->character = c[0];
              L->numPoints = 9+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=7; p[1]=2; p[2]=4; p[3]=13; p[4]=27; p[5]=32; p[6]=30; p[7]=21; p[8]=7; break;
     case 80: L->character = c[0];
              L->numPoints = 8+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=3; p[2]=6; p[3]=20; p[4]=33; p[5]=32; p[6]=17; p[7]=3; break;
     case 81: L->character = c[0];
              L->numPoints = 12+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=7; p[1]=2; p[2]=4; p[3]=13; p[4]=27; p[5]=32; p[6]=30; p[7]=21; p[8]=7; p[9]=-1; p[10]=17; p[11]=28; break;
     case 82: L->character = c[0];
              L->numPoints = 10+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=3; p[2]=6; p[3]=20; p[4]=33; p[5]=32; p[6]=17; p[7]=3; p[8]=17; p[9]=28; break;
     case 83: L->character = c[0];
              L->numPoints = 10+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=1; p[1]=7; p[2]=21; p[3]=29; p[4]=24; p[5]=10; p[6]=5; p[7]=13; p[8]=27; p[9]=33; break;
     case 84: L->character = c[0];
              L->numPoints = 5+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=6; p[1]=34; p[2]=20; p[3]=17; p[4]=14; break;
     case 85: L->character = c[0];
              L->numPoints = 6+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=6; p[1]=3; p[2]=7; p[3]=21; p[4]=31; p[5]=34; break;
     case 86: L->character = c[0];
              L->numPoints = 5+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=6; p[1]=10; p[2]=14; p[3]=24; p[4]=34; break;
     case 87: L->character = c[0];
              L->numPoints = 7+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=6; p[1]=3; p[2]=0; p[3]=17; p[4]=28; p[5]=31; p[6]=34; break;
     case 88: L->character = c[0];
              L->numPoints = 5+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=34; p[2]=17; p[3]=6; p[4]=28; break;
     case 89: L->character = c[0];
              L->numPoints = 5+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=6; p[1]=17; p[2]=34; p[3]=17; p[4]=14; break;
     case 90: L->character = c[0];
              L->numPoints = 10+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=6; p[1]=17; p[2]=34; p[3]=17; p[4]=10; p[5]=24; p[6]=17; p[7]=0; p[8]=14; p[9]=28; break;
     /* Zahlen */
     case 48: L->character = c[0];
              L->numPoints = 11+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=7; p[1]=2; p[2]=4; p[3]=13; p[4]=27; p[5]=32; p[6]=30; p[7]=21; p[8]=7; p[9]=17; p[10]=27; break;
     case 49: L->character = c[0];
              L->numPoints = 5+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=4; p[1]=19; p[2]=34; p[3]=31; p[4]=28; break;
     case 50: L->character = c[0];
              L->numPoints = 8+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=4; p[1]=13; p[2]=27; p[3]=32; p[4]=17; p[5]=0; p[6]=14; p[7]=28; break;
     case 51: L->character = c[0];
              L->numPoints = 11+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=1; p[1]=7; p[2]=21; p[3]=29; p[4]=24; p[5]=10; p[6]=24; p[7]=33; p[8]=27; p[9]=13; p[10]=5; break;
     case 52: L->character = c[0];
              L->numPoints = 5+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=21; p[1]=27; p[2]=3; p[3]=17; p[4]=31; break;
     case 53: L->character = c[0];
              L->numPoints = 9+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=21; p[2]=29; p[3]=30; p[4]=24; p[5]=3; p[6]=6; p[7]=20; p[8]=34; break;
     case 54: L->character = c[0];
              L->numPoints = 11+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=2; p[1]=10; p[2]=24; p[3]=30; p[4]=21; p[5]=7; p[6]=2; p[7]=5; p[8]=13; p[9]=27; p[10]=33; break;
     case 55: L->character = c[0];
              L->numPoints = 5+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=6; p[1]=20; p[2]=34; p[3]=17; p[4]=7; break;
     case 56: L->character = c[0];
              L->numPoints = 13+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=17; p[1]=2; p[2]=1; p[3]=14; p[4]=29; p[5]=30; p[6]=17; p[7]=4; p[8]=5; p[9]=20; p[10]=33; p[11]=32; p[12]=17; break;
     case 57: L->character = c[0];
              L->numPoints = 9+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=1; p[1]=14; p[2]=29; p[3]=33; p[4]=20; p[5]=5; p[6]=4; p[7]=17; p[8]=32; break;
     /* Sonderzeichen */
     case 32: L->character = c[0];
              L->numPoints = 0+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              break;
     case 46: L->character = c[0];
              L->numPoints = 1+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=14; break;
     case 33: L->character = c[0];
              L->numPoints = 5+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=14; p[1]=-1; p[2]=16; p[3]=18; p[4]=20; break;
     case 63: L->character = c[0];
              L->numPoints = 8+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=5; p[1]=20; p[2]=33; p[3]=32; p[4]=17; p[5]=16; p[6]=-1; p[7]=14; break;
     case 42: L->character = c[0];
              L->numPoints = 8+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=12; p[1]=24; p[2]=18; p[3]=10; p[4]=26; p[5]=18; p[6]=19; p[7]=17; break;
     case 43: L->character = c[0];
              L->numPoints = 5+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=3; p[1]=31; p[2]=17; p[3]=19; p[4]=15; break;
     case 45: L->character = c[0];
              L->numPoints = 3+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=3; p[1]=17; p[2]=31; break;
     case 61: L->character = c[0];
              L->numPoints = 7+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=2; p[1]=16; p[2]=30; p[3]=-1; p[4]=4; p[5]=18; p[6]=32; break;
     case 47: L->character = c[0];
              L->numPoints = 3+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=0; p[1]=17; p[2]=34; break;
     case 92: L->character = c[0];
              L->numPoints = 3+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=6; p[1]=17; p[2]=29; break;
     case 94: L->character = c[0];
              L->numPoints = 3+1;
              L->points = (int*)malloc(L->numPoints * sizeof(int));  p = L->points;
              p[0]=3; p[1]=20; p[2]=31; break;
   }
}

void getPoint(double *outX, double *outY, int *x0, int *y0, int *nextPointOfLetter, int *nextLetter, letter_t *L){
   if(*nextPointOfLetter+1 == L->numPoints){
     *outX = *x0;
     *outY = INVALID_MEASUREMENT;  // letzter "Punkt" verhindert verbiden zweier aufeinander folgender Buchstaben
     *nextPointOfLetter = 0;
     *nextLetter += 1;
     *x0 += 5 + 1;
   } else {
     if(L->points[*nextPointOfLetter]==-1){
       *outX = *x0;
       *outY = INVALID_MEASUREMENT;  // Sprung der Linie im Buchstaben, zb bei "!"
       *nextPointOfLetter += 1;
     } else {
       *outX = (double)(*x0 + L->points[*nextPointOfLetter] / 7);
       *outY = (double)(*y0 + L->points[*nextPointOfLetter] % 7);
       *nextPointOfLetter += 1;
     }
   }
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
  /* ii is used for loop iterations */
  int temp, i, imyproblemSize = iproblemSize;
  /* cast void* pointer */
  mydata_t * pmydata = (mydata_t *) mdpv;

  IDL(3, printf("problemSize=%d\n",imyproblemSize));
  
  /* check wether the pointer to store the results in is valid or not */
  if (dresults == NULL) return 1;

  /* prepare dresults */
  for(i=0; i<allLetter; i++) dresults[i+1] = INVALID_MEASUREMENT;
  
  temp = (int)(pmydata->str[nextLetter]);
  /* generate new line when found character "_" */
  if(temp==95){
    letterFirstPosX = 1;
    letterFirstPosY -= 10;
    nextLetter++;
    temp = (int)(pmydata->str[nextLetter]);
  }
  /* Buchstaben */
  if(temp>=65 && temp<=65+26){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[temp-65+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[temp-65]));
  }
  i = 26;
  if(temp==32){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==33){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }

#ifdef zuviel
  /* Zahlen */
  if(temp>=48 && temp<=48+10){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[temp-48+26+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[temp-48+26]));
  }
  /* Sonderzeichen */
//  i = 36; /* Buchstaben+Zahlen */
//  if((temp==32 || i++) || (temp==46 || i++) || (temp==33 || i++) || (temp==63 || i++) || (temp==42 || i++) || (temp==43 || i++)
//                        || (temp==45 || i++) || (temp==61 || i++) || (temp==47 || i++) || (temp==92 || i++) || (temp==94 || i++)){
//    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
//    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
//  }

  i = 36; /* Buchstaben+Zahlen */
  if(temp==32){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==46){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==33){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==63){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==42){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==43){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==45){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==61){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==47){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==92){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
  i++;
  if(temp==94){
    IDL(3, printf("char=%c\n",pmydata->str[nextLetter]));
    getPoint(&(dresults[0]), &(dresults[i+1]), &letterFirstPosX, &letterFirstPosY, &letterNextPos, &nextLetter, &(pmydata->myletters[i]));
  }
#endif
  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   int i;
   mydata_t * pmydata = (mydata_t*)mdpv;
   if (pmydata){
     for(i=0; i<allLetter; i++) free(pmydata->myletters[i].points);
     free(pmydata->myletters);
     free(pmydata);
   }
   return;
}

