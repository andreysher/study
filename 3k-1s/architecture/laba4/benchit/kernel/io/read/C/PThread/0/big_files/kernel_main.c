/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: kernel_main.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/io/read/C/PThread/0/big_files/kernel_main.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "interface.h"

#include "tpool.h"
#include "iobigread.h"
#include "iobig_readfct.h"
#include "eval.h"
#include "iocreate.h"
#include "ioremove.h"

	tpool_t threadpool;

/*  wrapper for 
   void readfiles(problemSize, global->maxdeep, btime+i, etime+i)
   because pthreads take only one argument
*/
void thread_readfile(thread_arg_wrapper_t *taw)
{
	readfiles(taw->problemSize, taw->global->maxdeep, taw->btime, taw->etime);
}



int max_problemSize(iods * pmydata)
{
   int maxps = 0;
   double FILESIZE = pmydata->FILESIZE, DISKSPACE = pmydata->DISKSPACE;
   
	/*decades beetween filesize and diskspace -> 
	  reduplication: (set{1,5,10,50,...)}*/
        maxps=((int)(log10(DISKSPACE))-(int)(log10(FILESIZE)))*2;
	/*if filesize is bigger then 5*10^xmin -> one less*/
        if ((2*FILESIZE)>(pow(10,(double)((int)(log10(FILESIZE))+1)))) maxps--;
	/*if diskspace is more then 5*10^xmax -> one more*/
        if ((2*(DISKSPACE/1000000))>=(pow(10,(double)((int)(log10(DISKSPACE/1000000))+1)))) maxps++;
	/*if diskspace is not part of the set -> one more (diskspace itself)*/
        if ((DISKSPACE/1000000)!=(pow(10,(double)((int)(log10(DISKSPACE/1000000)))))) maxps++;
	/*if diskspace a part of the "5-set" (filesize=5*10^x) -> one less*/
	if ((DISKSPACE/1000000)==5*(pow(10,(double)((int)(log10(DISKSPACE/1000000)))))) maxps--;
	return maxps;
}
	

/**  The implementation of the bi_getinfo from the BenchIT interface.
 *   Here the infostruct is filled with information about the
 *   kernel.
 *   @param infostruct  a pointer to a structure filled with zeros
 */
void bi_getinfo(bi_info * infostruct)
{
   iods * penv;
   penv = (iods *) malloc(sizeof(iods));

   /* get environment variables for the kernel */
   evaluate_environment(penv);
   infostruct->codesequence = bi_strdup("mkdir binary-directory-tree; read leaves (=files)");
   infostruct->kerneldescription = bi_strdup("io kernel reading large files using PThreads");
   infostruct->xaxistext = bi_strdup("used diskspace/Byte");
   infostruct->num_measurements = max_problemSize(penv);
   infostruct->num_processes = 1;
   infostruct->num_threads_per_process = (penv->NUMCHANNELS * penv->CHANNELFACTOR);
   infostruct->kernel_execs_mpi1 = 0;
   infostruct->kernel_execs_mpi2 = 0;
   infostruct->kernel_execs_pvm = 0;
   infostruct->kernel_execs_omp = 0;
   infostruct->kernel_execs_pthreads = 1;
   infostruct->numfunctions = 1;

   /* allocating memory for y axis texts and properties */
   allocYAxis(infostruct);
   /* setting up y axis texts and properties */
      infostruct->yaxistexts[0] = bi_strdup("Byte/s");
      infostruct->selected_result[0] = SELECT_RESULT_HIGHEST;
      infostruct->base_yaxis[0] = 10; //logarythmic axis 10^x
      infostruct->legendtexts[0] = bi_strdup("time in s");
 
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
	createFiles();
	
	iods * pmydata;
	const char *path, *tmpheader;
	char *fileheader, *destination;
        double dsz, fsz, ds, fs;
        long i=0;
	FILE *fp;


   pmydata = (iods*)malloc(sizeof(iods));
   if (pmydata == 0)
   {
      fprintf(stderr, "Allocation of structure iods failed\n"); fflush(stderr);
      exit(127);
   }
   evaluate_environment(pmydata);

   
	tmpheader=pmydata->TMPHEADER;
	if(tmpheader==NULL) {printf ("\n Cant get header of tmp-file\n"); exit(127);}
	path=pmydata->DISKPATH;
	if(path==NULL) { printf("\nCant get path for writing!\n"); exit(127); }

	dsz=pmydata->DISKSPACE;
	fsz=pmydata->FILESIZE;
	dsz=dsz/fsz;


	pmydata->startpath=malloc(4096*sizeof(char));
	getcwd(pmydata->startpath, 4096);

	pmydata->path=malloc(4096*sizeof(char));
	if(pmydata->path==NULL) { printf("\nCant get memory for pathstring!\n"); exit(127); }
	strncpy(pmydata->path, pmydata->DISKPATH, strlen(pmydata->DISKPATH));
	fileheader=malloc(1024*sizeof(char));
	if(fileheader==NULL) { printf("\nCant get memory to compare fileheaders!\n"); exit(127); }
	destination=malloc(4096*sizeof(char));
	if(destination==NULL) { printf("\nCant get memory to find tmp-file!\n"); exit(127); }
	sprintf(destination, "%siobig.tmp", pmydata->path);
	for(;;)
	{
		fp=fopen(destination, "r");
		if(fp==NULL) i++;
		else
		{
			fscanf(fp, "%[^\n]", fileheader); fgetc(fp);
			fscanf(fp, "%[^\n]", pmydata->path); fgetc(fp);
			fscanf(fp, "%lf %lf", &ds, &fs);
			fclose(fp);
			if(fs==fsz && (long)ds==(long)dsz && !(strcmp(tmpheader, fileheader))) break;
		}
		sprintf(destination, "%siobig%ld.tmp", path, i);
		if(i>1024) break;
	}

	printf("\nFound fitting tmp-file.\n");
	printf("Performing benchmark in %s\n", pmydata->path);

	i=(long)(ds-1);
	pmydata->maxdeep=(i>>pmydata->POTFILPERDIR==0) ? 0 : (long)(log((double)(i>>pmydata->POTFILPERDIR))/log(2))+1;
   
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
 /* ii is used for loop iterations */
  myinttype ii = 0, imyproblemSize = (myinttype) iproblemSize;
  /* cast void* pointer */
  iods * pmydata = (iods *) mdpv;

	double *btime, *etime, ds, fs, min, max;
	long i, j, tawsize;
	int num_threads, queue_length, mr, nc, cf, rs, fp;
	thread_arg_wrapper_t *taw;

	/*ds = diskspace/MB*(MB/filesize) -> number of created files*/
	ds=pmydata->DISKSPACE;
	fs=pmydata->FILESIZE;
	mr=pmydata->MAXREPEAT;
	nc=pmydata->NUMCHANNELS; 
	cf=pmydata->CHANNELFACTOR;
	rs=pmydata->REPEATSTOP;
	fp=pmydata->FILESPERTHREAD;
	ds=ds/fs;

	
/* check wether the pointer to store the dresults in is valid or not */
  if (dresults == NULL) return 1;

/*recalculation of the "simulated ds" from the given imyproblemSize 
  (imyproblemsiz is converted into set{1,5,10,50,100,...} - begin*/
	if((2*fs)>(pow(10,(double)((long)(log10(fs))+1)))) imyproblemSize=(pow(10,(double)((long)(log10(fs))+1)));
	else imyproblemSize=(pow(10,(double)((long)(log10(fs))+1))/2);
	iproblemSize--;
	while(iproblemSize>0)
	{
		if((double)((long)(log10(imyproblemSize)))==log10(imyproblemSize)) imyproblemSize=5*imyproblemSize;
		else imyproblemSize=2*imyproblemSize;
		iproblemSize--;
	}
/*end*/

	
	printf("\ngiven problemSize=%d",iproblemSize);
	printf("\ncalc. problemSize=%d",imyproblemSize);
	printf("\npath=%s",pmydata->path);
	printf("\nstartpath=%s",pmydata->startpath);
	printf("\nDISKPATH=%s",pmydata->DISKPATH);
	printf("\n");

	/*adjusting "simulated ds" to conform with fs*/
	imyproblemSize=(double)((long)(imyproblemSize/fs));
	/*maximum imyproblemSize = available ds*/
	imyproblemSize=imyproblemSize>(ds) ? (ds) : imyproblemSize;

	printf("measurement with %i MB of simulated ds\n",
	       (((int)(imyproblemSize*fs)) / (1048576)));

	/*array(s) for time measurement*/
	btime=calloc((int)(ds*mr), sizeof(double));
	if (btime==NULL) { printf("\nCant get memory for time measurement!\n"); exit(127); } 
	etime=calloc((int)(ds*mr), sizeof(double));
	if (etime==NULL) { printf("\nCant get memory for time measurement!\n"); exit(127); }

	/*changing to the created paths -> important for the following*/
	if(chdir(pmydata->path)) { printf("Cant change directory to %s\n", pmydata->path); exit(127); }

	num_threads = nc * cf ;
	queue_length = num_threads*6;
	tpool_init(&threadpool, num_threads, queue_length,0);
	if ((long)(ds*mr) > (long)(rs*imyproblemSize))
		tawsize = (long)(ds*mr);
	else tawsize = (long)(rs*imyproblemSize);
	taw = calloc(tawsize, sizeof(thread_arg_wrapper_t));

	for(i=0;i<tawsize;i++)
	{
	     taw[i].problemSize = (long)imyproblemSize;
	     taw[i].btime = &btime[i]; 
	     taw[i].etime = &etime[i];
	     taw[i].global = (iods *)pmydata;	     
	}

	i=0;

	/*per call the kernel reads a maximum of (written)files*mr or rs*number of the regarded files*/
	while(i<(long)(ds*mr) && i<(long)(rs*imyproblemSize))
	{
		/*calling the read function (=the actual measurement function)*/
		tpool_add_work(threadpool, &thread_readfile,(void *)&taw[i]); 
		i++;
	}
	tpool_destroy(threadpool, 1);

	/*x-axis = simulated ds*/
	dresults[0]=(double)(imyproblemSize*fs);

	/*time calculation*/
	min=btime[0];
	max=etime[0];
	for(j=0;j<i;j++)
	{
		min=btime[j]<min ? btime[j] : min;
		max=etime[j]>max ? etime[j] : max;
	}

	/*y-axis = average transfer rate*/
	dresults[1]=(double)(i)*fs*fp/(max-min);
	
	if(chdir(pmydata->startpath)) printf("\nCant change to directory %s\n. You will probably not get an output-file.", pmydata->startpath);
	
	free(taw);
	free(btime);
	free(etime);

/*
  dstart = bi_gettime(); 
  dres = simple(&imyproblemSize); 
  dend = bi_gettime();


  dtime = dend - dstart;
  dtime -= dTimerOverhead;
      
  if(dtime < dTimerGranularity) dtime = INVALID_MEASUREMENT;

  dresults[0] = (double)imyproblemSize;
  dresults[1] = dtime;
*/

  return 0;
}

/** Clean up the memory
 */
void bi_cleanup(void* mdpv)
{
   iods * pmydata = (iods *)mdpv;
   if (pmydata->path) free(pmydata->path);
   if (pmydata->startpath) free(pmydata->startpath);
   if (pmydata) free(pmydata);
   removeFiles();
   return;
}
