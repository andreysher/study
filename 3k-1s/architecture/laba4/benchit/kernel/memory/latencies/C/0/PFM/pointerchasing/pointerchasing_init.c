/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: pointerchasing_init.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/memory/latencies/C/0/PFM/pointerchasing/pointerchasing_init.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "interface.h"
#include "pointerchasing.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <perfmon/pfmlib.h>

#include <perfmon/perfmon.h>
#include <perfmon/pfmlib_itanium2.h>


#define MAX_EVT_NAME_LEN        256
#define NUM_PMCS PFMLIB_MAX_PMCS
#define NUM_PMDS PFMLIB_MAX_PMDS

#ifndef MIN_ACCESS_LENGTH
#define MIN_ACCESS_LENGTH (2048)
#endif

#ifndef MAX_ACCESS_LENGTH
#define MAX_ACCESS_LENGTH (1024*1024)
#endif

#ifndef ACCESS_STEPS
#define ACCESS_STEPS (100)
#endif

#ifndef NUMBER_OF_JUMPS
#define NUMBER_OF_JUMPS (4000000)
#endif


unsigned int random_number(unsigned long max);
void make_linked_memory(void *mem, long count);
void init_global_vars(void);

long minlength, maxlength, accessstride, numjumps;
double dMemFactor;
long nMeasurements;

int NUM_COUNTERS;
char sCounters[10][100];


void bi_getinfo(bi_info* infostruct){
  int i, l;
  char buf[200], *s;
  int events[10];

  init_global_vars();

	
  /*infostruct->kernelstring=bi_strdup("Random Memory Access");*/
  infostruct->codesequence=bi_strdup("for i=1,N#  var=memory[random(0..size)]#");
  infostruct->xaxistext=bi_strdup("Accessed Memory in Byte");
  
  infostruct->numfunctions= 1+ NUM_COUNTERS;
  infostruct->num_measurements=nMeasurements;
  
  /* allocating memory for y axis texts and properties */
  allocYAxis(infostruct);
  
  for (i=0; i< infostruct->numfunctions; i++){
  		infostruct->selected_result[i] = SELECT_RESULT_AVERAGE;
  		infostruct->yaxistexts[i]=bi_strdup("");
		  infostruct->log_yaxis[0]=0;
  		infostruct->base_yaxis[0]=0.0;
  }
		
  infostruct->yaxistexts[0]=bi_strdup("s");
  infostruct->log_xaxis=1;
  infostruct->base_xaxis=2.0;
  
	if (pfm_initialize() != PFMLIB_SUCCESS) {
		fprintf(stderr, "cannot initialize libpfm\n");
		exit(0);
	}

}

void init_global_vars() {
    
	char *envir, *p, *q;
	int i;

	IDL(3,printf("Init global variables ... "));
	envir=bi_getenv("MIN_ACCESS_LENGTH",1);
	minlength=(envir != NULL) ? 1024*atoi(envir) : MIN_ACCESS_LENGTH;
	if(minlength==0) {
		minlength=MIN_ACCESS_LENGTH;
	}
	envir=NULL;
	envir=bi_getenv("MAX_ACCESS_LENGTH",1);
	maxlength=(envir != NULL) ? 1024*atoi(envir) : MAX_ACCESS_LENGTH;
	if(maxlength==0) {
		maxlength=MIN_ACCESS_LENGTH;
	}
	envir=NULL;
	envir=bi_getenv("ACCESS_INCREMENT",0); /*in kB */
	if (envir != NULL) {
		i = atoi(envir);
		nMeasurements = (maxlength-minlength)/(i*1024);
		dMemFactor = -i;	/* if <0 : linear measurement */
	}
	else {
		envir=bi_getenv("ACCESS_STEPS",1);
		if (envir != NULL) {
			nMeasurements = (envir != 0) ? atoi(envir) : ACCESS_STEPS;
			dMemFactor =((double)maxlength)/((double)minlength);
			dMemFactor = pow(dMemFactor, 1.0/((double)nMeasurements-1));
		}
	}		
		
	envir=NULL;
	envir=bi_getenv("NUMBER_OF_JUMPS",1);
	numjumps=(envir != NULL) ? atoi(envir) : NUMBER_OF_JUMPS;
	if(numjumps==0) {
		numjumps=NUMBER_OF_JUMPS;
	}
	envir = bi_getenv("PAPI_COUNTERS",1);
	p = envir;
	NUM_COUNTERS = 0;
	while (p) {
		p = strchr(p, ',');
		if (p) p++; 
		NUM_COUNTERS++;
	}
	/*sCounters = (char*) malloc(sizeof(char*) * NUM_COUNTERS);*/
	i = 0;
	p = envir;
	for (i=0; i<NUM_COUNTERS; i++) {	
		q = strchr(p, ',');
		if (q) {		
			strncpy(sCounters[i], p, (int)(q-p));
			sCounters[i][q-p+1]=0;
			p = ++q;
		}	
		else
			strcpy(sCounters[i], p);	
	}
	IDL(3,printf("done\n"));
}

BI_GET_CALL_OVERHEAD_FUNC((),jump_around(NULL,0));


void *bi_init(int problemSizemax){
  void *mem;

  IDL(3, printf("Enter init ... "));
  mem=malloc(maxlength);
  if (mem==NULL){
    printf("No more core, need %.3f MByte\n", 
	   (double)maxlength);
    exit(127);
  }
  IDL(3, printf("allocated %.3f MByte\n",
		(double)maxlength));
  return (mem);
}

void bi_cleanup(void *mcb){
  free(mcb);
  return;
}
