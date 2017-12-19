#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
/* main posix header */
#include <unistd.h>

#include "stringlib.h"
#include "bienvhash.h"
#include "output.h"
#include "gnuWriter.h"

#include "bitWriter.h"

double BI_INFINITY = 0.0; /**< default values to be used if values are to big, set when starting main()*/
double BI_NEG_INFINITY = 0.0; /**< default values to be used if values are to small, set when starting main()*/

/*
 * will contain data for all axis (mins, maxs, ...)
 */
static axisdata xdata, *ydata = NULL, ydata_global;

int initResults(bi_info theInfo) {
	/* setting infinities */
	BI_INFINITY = pow(2.0, 1023.0);
	BI_NEG_INFINITY = -BI_INFINITY;
	/* information for the y-axis */
	ydata = (axisdata*) calloc((size_t) theInfo.numfunctions, sizeof(axisdata));
	if (ydata == 0)
		return 1;
	/* fill them with bytes 0 */
	(void) memset(&xdata, 0, sizeof(axisdata));
	(void) memset(&ydata_global, 0, sizeof(axisdata));
	/* standard name for axis */
	xdata.name = 'x';
	ydata_global.name = 'y';
	int i;
	for (i = 0; i < theInfo.numfunctions; ++i)
		ydata[i].name = 'y';
	return 0;
}

/*!@brief Separate the components of the kernelname.
 *
 * @param[out] kernelstring The complete name of the kernel.
 * @param[out] language Programming language of the kernel.
 * @param[out] libraries The software libraries used by the kernel.
 * @param[out] numLibraries The count of used libraries.
 */
static int getKernelNameInfo(char **kernelstring, char **language, char ***libraries, int *numLibraries) {
	/*
	 get the kernel name (numerical.matmul.C.0.0.double)
	 */
	char *kernelname = bi_getenv("BENCHIT_KERNELNAME", 1);
	/*variables:
	 id:which part of the kernelname
	 ps,pe:used for start and end of parts of the kernelname
	 klen:kernelname-length
	 pnum:number of parallel libraries
	 onum:number of other libraries
	 i:at the end of this procedure used for loops, which lib is written
	 */
	int id = 0, ps = -1, pe = -1, klen = -1, pnum = 0, onum = 0, i = 0;
	/*
	 buffer for library names
	 */
	char lbuf[100000];
	/*
	 parallel and other libs (limited to 10), do you really want to use more?
	 */
	char *plibs[10], *olibs[10];
	/* fills lbuf with 0s (NOT '0's) */
	memset(lbuf, 0, 100000);
	/* get the length of e.g. numerical.matmul.C.0.0.double */
	klen = (int) strlen(kernelname);
	/* set the first return value for the kernelname as copy from kernelname */
	*kernelstring = bi_strdup(kernelname);
	/* walk through the categories */
	while ((pe = indexOf(kernelname, '.', ps + 1)) >= 0) {
		id++;
		/* get kernel source language */
		if (id == 3) {
			substring(kernelname, lbuf, ps + 1, pe);
			*language = bi_strdup(lbuf);
		}
		/* get libs for parallel programming */
		else if (id == 4) {
			if (ps + 1 < klen && kernelname[ps + 1] != '0') {
				int nextDot = indexOf(kernelname, '.', ps + 1);
				int lbufpos = 0;
				if (nextDot > ps + 1 && nextDot + 1 < klen) {
					/* nextDot is index of category end in kernelname */
					for (i = ps + 1; i < nextDot; i++) {
						if (kernelname[i] == '-' && kernelname[i + 1] == '-') {
							/* found separator */
							/* end the last found item of parallel libs */
							lbuf[lbufpos] = '\0';
							/* reset the beginning of the buffer */
							/* (the next item starts also at 0 and will overwrite the other one) */
							lbufpos = 0;
							/* add lib to list */
							plibs[pnum++] = bi_strdup(lbuf);
						} else {
							/* add next character to the next parallel lib-name */
							lbuf[lbufpos++] = kernelname[i];
						}
					}
					/* end the last existing item of parallel libs */
					lbuf[lbufpos] = '\0';
					/* add lib to list */
					plibs[pnum++] = bi_strdup(lbuf);
				}
			}
		}
		/* get other libs */
		else if (id == 5) {
			if (ps + 1 < klen && kernelname[ps + 1] != '0') {
				int nextDot = indexOf(kernelname, '.', ps + 1);
				int lbufpos = 0;
				if (nextDot > ps + 1 && nextDot + 1 < klen) {
					/* nextDot is index of category end in kernelname */
					for (i = ps + 1; i < nextDot; i++) {
						if (kernelname[i] == '-' && kernelname[i + 1] == '-') {
							/* found separator */
							/* end last item of other libs */
							lbuf[lbufpos] = '\0';
							/* reset the beginning of the buffer */
							lbufpos = 0;
							/* add lib to list */
							olibs[onum++] = bi_strdup(lbuf);
						} else {
							/* add next character to the next parallel lib-name */
							lbuf[lbufpos++] = kernelname[i];
						}
					}
					/* end the last existing item of parallel libs */
					lbuf[lbufpos] = '\0';
					/* add lib to list */
					olibs[onum++] = bi_strdup(lbuf);
				}
			}
		}
		/* new start '.' is old end '.' */
		ps = pe;
	} /* end of walk through categories */
	freeCheckedC(&kernelname);
	/* total libraries are both parallel and others */
	*numLibraries = pnum + onum;
	/* if we dont have any libraries, we still have one: NONE ;) */
	if (*numLibraries == 0)
		*numLibraries = 1;
	/* getting memory for libraries */
	*libraries = ((char **) calloc((size_t) *numLibraries, sizeof(char *)));
	if (*libraries == 0) {
		printf(" [FAILED] (no more memory for libs)\n");
		return 1;
	}
	/* add the libs */
	id = 0;
	/* first the parallel libs */
	for (i = 0; i < pnum; i++) {
		(*libraries)[id++] = plibs[i];
	}
	/* then the other libs */
	for (i = 0; i < onum; i++) {
		(*libraries)[id++] = olibs[i];
	}
	return 0;
}

/*!****************************************************************************
 * Analyzing results (Getting Min, Max)
 */
static void analyse_results(bi_info theInfo, double* results) {
	/* initialize mins and maxs */
	xdata.min = BI_INFINITY;
	xdata.max = BI_NEG_INFINITY;
	int i, j;
	for (j = 0; j < theInfo.numfunctions; j++) {
		ydata[j].min = BI_INFINITY;
		ydata[j].max = BI_NEG_INFINITY;
	}

	int dataPointCt = theInfo.numfunctions + 1;
	/* find mins and maxs for x-axis and y-axis within the results */
	for (i = 0; i < theInfo.num_measurements; i++) {
		int curXIndex = i * dataPointCt;
		double curX = results[curXIndex];
		if (curX < 0)
			continue; /* <0 : invalid or not measured (timeout/abort) */

		/* set min for xaxis */
		if ((curX < xdata.min) && (curX > BI_NEG_INFINITY))
			xdata.min = curX;
		/* set max for xaxis */
		if ((curX > xdata.max) && (curX < BI_INFINITY))
			xdata.max = curX;
		for (j = 0; j < theInfo.numfunctions; j++) {
			double curY = results[curXIndex + j + 1];
			if (curY == INVALID_MEASUREMENT)
				continue;
			/* set min for yaxis[j] */
			if (curY < ydata[j].min && curY > BI_NEG_INFINITY)
				ydata[j].min = curY;
			/* set max for yaxis[j] */
			if (curY > ydata[j].max && curY < BI_INFINITY)
				ydata[j].max = curY;
		}
	}

	/* Get global minimum and maximum for y-axis */
	ydata_global.min = BI_INFINITY;
	ydata_global.max = BI_NEG_INFINITY;
	for (i = 0; i < theInfo.numfunctions; ++i) {
		if (ydata[i].min < ydata_global.min)
			ydata_global.min = ydata[i].min;
		if (ydata[i].max > ydata_global.max)
			ydata_global.max = ydata[i].max;
	}

	/* adjust min and max of x-axis if we only have one measurement */
	if (xdata.min == xdata.max) {
		xdata.max *= 1.1;
		xdata.min *= 0.9;
		if (xdata.min == xdata.max)
			xdata.max += 1.0;
	}
	/* adjust min and max of y-axis if we only have one measurement */
	if (ydata_global.min == ydata_global.max) {
		ydata_global.max *= 1.1;
		ydata_global.min *= 0.9;
		if (ydata_global.min == ydata_global.max)
			ydata_global.max += 1.0;
	}
}

/*!@brief Gets the best (as defined by kernel) results from the data
 *
 * @param[in] theInfo The info struct from the kernel
 * @param[in] results All results (x|(func1|func2|...)*repeatCt)
 * @param[in] repeatCt Number of values per function
 */
static double* getBestResults(bi_info theInfo, double* results, int repeatCt) {
	int i, j, k;
	//Data points per measurement including x
	int dataPointCt = theInfo.numfunctions * repeatCt + 1;
	int dataPointCtBest = theInfo.numfunctions + 1;
	double* bestValues = malloc((size_t) theInfo.num_measurements * (size_t) dataPointCt * sizeof(double));
	if (bestValues == NULL )
		return NULL ;
	for (i = 0; i < theInfo.num_measurements; i++) {
		int curXIndex = i * dataPointCt;
		double curX = results[curXIndex];
		bestValues[i * dataPointCtBest] = curX;
		//Check for no or invalid measurement
		if (curX < 0)
			continue;
		for (j = 0; j < theInfo.numfunctions; j++) {
			double bestY = INVALID_MEASUREMENT;
			int validCt = 0;
			for (k = 0; k < repeatCt; k++) {
				double curY = results[curXIndex + j * repeatCt + k + 1];
				if (curY == INVALID_MEASUREMENT)
					continue;
				validCt++;
				if (validCt == 1)
					bestY = curY;
				else {
					if (theInfo.selected_result[j] == SELECT_RESULT_HIGHEST) {
						if (curY > bestY)
							bestY = curY;
					} else if (theInfo.selected_result[j] == SELECT_RESULT_LOWEST) {
						if (curY < bestY)
							bestY = curY;
					} else if (theInfo.selected_result[j] == SELECT_RESULT_AVERAGE) {
						bestY = (bestY * (validCt - 1) + curY) / validCt;
					}
				}
			}
			bestValues[i * dataPointCtBest + j + 1] = bestY;
		}
	}
	return bestValues;
}

/*!@brief Writes the data to file
 *
 * @param[in] file Open file pointer
 * @param[in] data The data as x|val1|val2|val3|...
 * @param[in] ct Number of values (x-axis)
 * @param[in] offset Stride (number of values on y-axis + 1 (for x value))
 */
static void writeDataToFile(FILE *file, double* data, int ct, int offset) {
	int i, j;
	for (i = 0; i < ct; i++) {
		if (data[i * offset] < 0){
			if(data[i * offset] != INVALID_MEASUREMENT)
				printf("\n Warning: Problemsize %f < 0.0 - ignored", data[i * offset]);
			continue;
		}
		for (j = 0; j < offset; j++) {
			if (data[i * offset + j] != INVALID_MEASUREMENT)
				fprintf(file, "%g\t", data[i * offset + j]);
			else
				fprintf(file, "-\t");
		}
		fprintf(file, "\n");
	}
}

static int writeBitFile(char* fileName, bi_info theInfo, double* results, struct tm* curTime, int standalone) {
	char *kernelString, *language, **libraries;
	int numLibraries;
	if (getKernelNameInfo(&kernelString, &language, &libraries, &numLibraries) != 0)
		return 1;

	FILE *bitFile = fopen(fileName, "w");
	if (bitFile == 0) {
		printf(" [FAILED]\nBenchIT: could not create output-file \"%s\"\n", fileName);
		return 127;
	}
	bi_fprintf(bitFile, "# BenchIT-Resultfile\n#\n");
	bi_fprintf(bitFile, "# feel free to fill in more architectural information\n");
	bi_fprintf(bitFile, "#########################################################\n");
	bi_fprintf(bitFile, "beginofmeasurementinfos\n");
	if (kernelString != 0)
		fprintf(bitFile, "kernelstring=\"%s\" \n", kernelString);
	else
		fprintf(bitFile, "kernelstring=\n");
	char timeBuf[35];
	(void) strftime(timeBuf, (size_t) sizeof(timeBuf), "%b %d %H:%M:%S %Y", curTime);
	bi_fprintf(bitFile, "date=\"%s\"\n", timeBuf);
	char *p = bi_getenv("BENCHIT_NUM_CPUS", 0);
	if (p != 0) {
		bi_fprintf(bitFile, "numberofprocessorsused=%s\n", p);
		freeCheckedC(&p);
	} else
		bi_fprintf(bitFile, "numberofprocessorsused=\n");
	if (theInfo.num_processes != 0)
		bi_fprintf(bitFile, "numberofprocesses=%i\n", theInfo.num_processes);
	else
		bi_fprintf(bitFile, "numberofprocesses=\n");
	if (theInfo.num_threads_per_process != 0)
		bi_fprintf(bitFile, "numberofthreadsperprocesses=%i\n", theInfo.num_threads_per_process);
	else
		bi_fprintf(bitFile, "numberofthreadsperprocesses=\n");
	p = bi_getenv("BENCHIT_RUN_MAX_MEMORY", 0);
	if (p != 0) {
		bi_fprintf(bitFile, "memorysizeused=\"%s\"\n", p);
		freeCheckedC(&p);
	} else
		bi_fprintf(bitFile, "memorysizeused=\n");
	p = bi_getenv("BENCHIT_KERNEL_COMMENT", 0);
	if (p != 0) {
		bi_fprintf(bitFile, "comment=\"%s\"\n", p);
		freeCheckedC(&p);
	} else
		bi_fprintf(bitFile, "comment=\n");
	if (theInfo.kerneldescription)
		bi_fprintf(bitFile, "kerneldescription=\"%s\"\n", theInfo.kerneldescription);
	else
		bi_fprintf(bitFile, "kerneldescription=\n");
	if (language != 0)
		bi_fprintf(bitFile, "language=\"%s\"\n", language);
	else
		bi_fprintf(bitFile, "language=\n");
	p = bi_getenv("LOCAL_KERNEL_COMPILER", 0);
	if (p == 0)
		p = bi_getenv("BENCHIT_COMPILER", 0); /**<\brief old Variablename \deprecated */
	if (p == 0) {
		bi_fprintf(bitFile, "compiler=\n");
		bi_fprintf(bitFile, "compilerversion=\n");
		/* print c compiler + version if local kernel compiler is unknown */
		bi_fprintf(bitFile, "c_compiler=\"%s\"\n", bi_getenv("BENCHIT_COMPILETIME_CC", 0));
		bi_fprintf(bitFile, "c_compilerversion=\"%s\"\n", bi_getenv("BENCHIT_CC_COMPILER_VERSION", 0));
	} else {
		char *pCC = bi_getenv("BENCHIT_COMPILETIME_CC", 0);
		bi_fprintf(bitFile, "compiler=\"%s\"\n", p);
		if (pCC) {
			char *pCCV = bi_getenv("BENCHIT_CC_COMPILER_VERSION", 0);
			if (!strcmp(p, pCC)) {
				bi_fprintf(bitFile, "compilerversion=\"%s\"\n", pCCV);
			} else {
				char *pCompiler[8];
				int compilerCt = 4;
				pCompiler[0] = bi_getenv("BENCHIT_COMPILETIME_CXX", 0);
				pCompiler[1] = bi_getenv("BENCHIT_CXX_COMPILER_VERSION", 0);
				pCompiler[2] = bi_getenv("BENCHIT_COMPILETIME_F77", 0);
				pCompiler[3] = bi_getenv("BENCHIT_F77_COMPILER_VERSION", 0);
				pCompiler[4] = bi_getenv("BENCHIT_COMPILETIME_F90", 0);
				pCompiler[5] = bi_getenv("BENCHIT_F90_COMPILER_VERSION", 0);
				pCompiler[6] = bi_getenv("BENCHIT_COMPILETIME_F95", 0);
				pCompiler[7] = bi_getenv("BENCHIT_F95_COMPILER_VERSION", 0);
				int i;
				int found = 0;
				for (i = 0; i < compilerCt; i++) {
					if (pCompiler[i * 2]) {
						if (!strcmp(p, pCompiler[i * 2]))
							bi_fprintf(bitFile, "compilerversion=\"%s\"\n", pCompiler[i * 2 + 1]);
						found = 1;
						break;
					}
				}
				if (!found)
					bi_fprintf(bitFile, "compilerversion=\n");
				/* additionally print c compiler + version if kernel uses other language */
				bi_fprintf(bitFile, "c_compiler=\"%s\"\n", pCC);
				bi_fprintf(bitFile, "c_compilerversion=\"%s\"\n", pCCV);
				for (i = 0; i < compilerCt * 2; i++) {
					if (pCompiler[i])
						freeCheckedC(&pCompiler[i]);
				}
			}
			freeCheckedC(&pCC);
			freeCheckedC(&pCCV);
		} else {
			bi_fprintf(bitFile, "compilerversion=\n");
		}
	}
	p = bi_getenv("BENCHIT_COMPILERFLAGS", 0); /**<\brief old Variablename \deprecated */
	if (p == 0)
		p = bi_getenv("LOCAL_KERNEL_COMPILERFLAGS", 0);
	if (p != 0) {
		bi_fprintf(bitFile, "compilerflags=\"%s\"\n", p);
		freeCheckedC(&p);
	} else
		bi_fprintf(bitFile, "compilerflags=\n");

	/* print C compiler infos */
	/* print_C_Compiler_information_to_file(bi_out,buf); */

	/* sizes of data types*/
	bi_fprintf(bitFile, "Size of basic data types:\n");
	bi_fprintf(bitFile, "  - sizeof(char)               : %2d Byte\n", (int) sizeof(char));
	bi_fprintf(bitFile, "  - sizeof(unsigned char)      : %2d Byte\n", (int) sizeof(unsigned char));
	bi_fprintf(bitFile, "  - sizeof(short)              : %2d Byte\n", (int) sizeof(short));
	bi_fprintf(bitFile, "  - sizeof(unsigned short)     : %2d Byte\n", (int) sizeof(unsigned short));
	bi_fprintf(bitFile, "  - sizeof(int)                : %2d Byte\n", (int) sizeof(int));
	bi_fprintf(bitFile, "  - sizeof(unsigned int)       : %2d Byte\n", (int) sizeof(unsigned int));
	bi_fprintf(bitFile, "  - sizeof(long)               : %2d Byte\n", (int) sizeof(long));
	bi_fprintf(bitFile, "  - sizeof(unsigned long)      : %2d Byte\n", (int) sizeof(unsigned long));
	bi_fprintf(bitFile, "  - sizeof(long long)          : %2d Byte\n", (int) sizeof(long long));
	bi_fprintf(bitFile, "  - sizeof(unsigned long long) : %2d Byte\n", (int) sizeof(unsigned long long));
	bi_fprintf(bitFile, "  - sizeof(float)              : %2d Byte\n", (int) sizeof(float));
	bi_fprintf(bitFile, "  - sizeof(double)             : %2d Byte\n", (int) sizeof(double));
	bi_fprintf(bitFile, "  - sizeof(long double)        : %2d Byte\n", (int) sizeof(long double));
	bi_fprintf(bitFile, "  - sizeof(void*)              : %2d Byte\n", (int) sizeof(void*));

	/* write kernellibraries*/
	int i;
	for (i = 0; i < numLibraries; i++) {
		if (libraries[i])
			bi_fprintf(bitFile, "library%d=\"%s\"\n", i + 1, libraries[i]);
		else
			bi_fprintf(bitFile, "library%d=\"\"\n", i + 1);
	}

	/* check if kernel is parallel */
	i = 0;
	i += theInfo.kernel_execs_mpi1;
	i += theInfo.kernel_execs_mpi2;
	i += theInfo.kernel_execs_pvm;
	i += theInfo.kernel_execs_omp;
	i += theInfo.kernel_execs_pthreads;
	if (i)
		bi_fprintf(bitFile, "kernelisparallel=1\n");
	else
		bi_fprintf(bitFile, "kernelisparallel=0\n");

	/* write kernellibraries even if kernel has not declared them */
	if ((i != 0) && (numLibraries == 0)) {
		i = 0;
		if (theInfo.kernel_execs_mpi1 || theInfo.kernel_execs_mpi2) {
			bi_fprintf(bitFile, "library%d=\"MPI\"\n", ++i);
		}
		if (theInfo.kernel_execs_pvm) {
			bi_fprintf(bitFile, "library%d=\"PVM\"\n", ++i);
		}
		if (theInfo.kernel_execs_omp) {
			bi_fprintf(bitFile, "library%d=\"OpenMP\"\n", ++i);
		}
		if (theInfo.kernel_execs_pthreads) {
			bi_fprintf(bitFile, "library%d=\"PThreads\"\n", ++i);
		}
	}

	if (theInfo.additional_information) {
		/* additional information should be a comma separated list of key=value pairs*/
		int error = 0;
		char *start, *comma, *eq, *key, *value;

		start = bi_strdup(theInfo.additional_information);
		comma = start - 1;

		do {
			key = comma + 1;
			eq = strstr(key, "=");
			if (eq != NULL ) {
				value = eq + 1;
				if (key == eq) {
					printf("\n         WARNING: error parsing additional_information");
					error++;
					break;
				}
				comma = strstr(key, ",");
				eq[0] = '\0';
				if (comma != NULL ) {
					if (comma < eq) /*syntax error*/
					{
						printf("\n         WARNING: error parsing additional_information");
						error++;
						break;
					}
					comma[0] = '\0';
				}
				if (strlen(key) + strlen(value) < MAX_ADD_INFO_STR)
					bi_fprintf(bitFile, "%s=%s\n", key, value);
				else {
					error++;
					printf("\n         WARNING: additional_information contains substring that is too long");
				}
			} else /* no key=value pair -> use comment for arbitrary strings*/
			{
				printf("\n         WARNING: error parsing additional_information");
				error++;
				break;
			}
		} while (comma != NULL );
		freeCheckedC(&start);
		if (error)
			printf(
					"\n         errors occured during parsing additional_information \
	                     \n         make sure it only contains comma separated key=value pairs \
	                     \n         value may be empty, key has to have a min length of 1");
	}

	bi_fprintf(bitFile, "is3d=%i\n", theInfo.is3d);
	if (theInfo.codesequence != 0)
		bi_fprintf(bitFile, "codesequence=\"%s\"\n", theInfo.codesequence);
	else
		bi_fprintf(bitFile, "codesequence=\n");
	bi_fprintf(bitFile, "xinmin=%g\n", xdata.min);
	bi_fprintf(bitFile, "xinmax=%g\n", xdata.max);
	for (i = 0; i < theInfo.numfunctions; i++) {
		bi_fprintf(bitFile, "y%dinmin=%g\n", i + 1, ydata[i].min);
		bi_fprintf(bitFile, "y%dinmax=%g\n", i + 1, ydata[i].max);
	}
	bi_fprintf(bitFile, "endofmeasurementinfos\n");
	bi_fprintf(bitFile, "################################################\n");
	char *rootDir = NULL;
	if (!standalone) {
		IDL(2, printf("...OK\nWriting architectureinfos"));
		rootDir = bi_getenv("BENCHITROOT", 0);
		if (rootDir == 0) {
			printf(" [FAILED]\nBenchIT: BENCHITROOT not found in environment hash table.\n");
			fclose(bitFile);
			return 1;
		}
		p = bi_getenv("BENCHIT_NODENAME", 1);
		char archfileName[1000];
		sprintf(archfileName, "%s/LOCALDEFS/%s_input_architecture", rootDir, p);
		FILE *archFile = fopen(archfileName, "r");
		if (archFile == 0) {
			sprintf(archfileName, "%s/LOCALDEFS/PROTOTYPE_input_architecture", rootDir);
			archFile = fopen(archfileName, "r");
		}
		if (archFile == 0) {
			printf(" [FAILED]\nBenchIT: Cannot open input file \"input_architecture\"\n");
			fclose(bitFile);
			return 1;
		}
		bi_fprintf(bitFile, "beginofarchitecture\n# Architekturangaben\n");

		if (p == 0)
			bi_fprintf(bitFile, "nodename=\n");
		else {
			bi_fprintf(bitFile, "nodename=\"%s\"\n", p);
			freeCheckedC(&p);
		}

		p = bi_getenv("BENCHIT_HOSTNAME", 0);
		if (p == 0)
			bi_fprintf(bitFile, "hostname=\n");
		else {
			bi_fprintf(bitFile, "hostname=\"%s\" \n", p);
			freeCheckedC(&p);
		}
		char fileReadBuf[1000];
		while (fgets(fileReadBuf, sizeof(fileReadBuf) - 1, archFile) != (char *) 0) {
			bi_fprint(bitFile, fileReadBuf);
		}
		fclose(archFile);
		IDL(2, printf("...OK\nWriting displayinfos"));
	} else {
		bi_fprintf(bitFile, "beginofarchitecture\n# Architekturangaben\n");
		p = bi_getenv("BENCHIT_NODENAME", 1);
		if (p == 0)
			bi_fprintf(bitFile, "nodename=\n");
		else {
			bi_fprintf(bitFile, "nodename=\"%s\"\n", p);
			freeCheckedC(&p);
		}
		p = bi_getenv("BENCHIT_HOSTNAME", 0);
		if (p == 0)
			bi_fprintf(bitFile, "hostname=\n");
		else {
			bi_fprintf(bitFile, "hostname=\"%s\" \n", p);
			freeCheckedC(&p);
		}
		IDL(2, printf("Running standalone. Don't write architecture_info...OK\nWriting displayinfos"));
	}
	bi_dumpTableToFile(bitFile);
	IDL(2, printf("Writing displayinfos"));
	bi_fprintf(bitFile, "beginofdisplay\n");
	IDL(2, printf(" x-axis "));
	xdata.base = theInfo.base_xaxis;
	if (get_axis_properties(&xdata) != 0) {
		fclose(bitFile);
		return 127;
	}
	bi_fprintf(bitFile, "\nxoutmin=%g\n", xdata.plotmin);
	bi_fprintf(bitFile, "xoutmax=%g\n", xdata.plotmax);
	bi_fprintf(bitFile, "xaxisticks=%d\n", xdata.ticks);
	bi_fprintf(bitFile, "xaxislogbase=%g\n", xdata.base);
	if (theInfo.xaxistext == 0)
		bi_fprintf(bitFile, "xaxistext=\n");
	else
		bi_fprintf(bitFile, "xaxistext=\"%s\"\n", theInfo.xaxistext);
	/* now the y-axises */
	IDL(2, printf(" y-axis "));
	for (i = 0; i < theInfo.numfunctions; i++) {
		ydata[i].base = theInfo.base_yaxis[i];
		if (i == 0)
			ydata_global.base = theInfo.base_yaxis[i];
		if (get_axis_properties(&(ydata[i])) != 0) {
			fclose(bitFile);
			return 127;
		}
		IDL(2, printf(" . "));
		bi_fprintf(bitFile, "\ny%doutmin=%g\n", i + 1, ydata[i].plotmin);
		bi_fprintf(bitFile, "y%doutmax=%g\n", i + 1, ydata[i].plotmax);
		bi_fprintf(bitFile, "y%daxisticks=%d\n", i + 1, ydata[i].ticks);
		bi_fprintf(bitFile, "y%daxislogbase=%g\n", i + 1, ydata[i].base);
		/* write axistexts */
		if (theInfo.yaxistexts[i] == 0)
			bi_fprintf(bitFile, "y%daxistext=\n", i + 1);
		else
			bi_fprintf(bitFile, "y%daxistext=\"%s\"\n", i + 1, theInfo.yaxistexts[i]);
	}

	bi_fprintf(bitFile, "numfunctions=%d\n\n", theInfo.numfunctions);
	bi_fprintf(bitFile, "## Fuer Architektur- oder Messmerkmal : abc\n#\n");
	bi_fprintf(bitFile, "# displayabc=1                                           # Steuervariable\n");
	bi_fprintf(bitFile, "# #Textfeld-Eigenschaften des date-Strings               # Kommentar\n");
	bi_fprintf(bitFile, "# tabc=\"ABC-Merkmal: 3 Tm\"                               # Text: Wert\n");
	bi_fprintf(bitFile, "# xabc=                                                  # x-Position\n");
	bi_fprintf(bitFile, "# yabc=                                                  # y-Position\n");
	bi_fprintf(bitFile, "# fonttypabc=                                            # Schriftart\n");
	bi_fprintf(bitFile, "# fontsizeabc=                                           # Schriftgroesse\n");
	for (i = 0; i < theInfo.numfunctions; i++) {
		if ((i == 0) && (theInfo.legendtexts == 0)) {
			(void) fprintf(stderr, "BenchIT: info->legendtexts==NULL\n");
			(void) fflush(stderr);
			fclose(bitFile);
			return 127;
		}
		if (theInfo.legendtexts[i] == 0)
			bi_fprintf(bitFile, "tlegendfunction%d=\n", i + 1);
		else
			bi_fprintf(bitFile, "tlegendfunction%d=\"%s\"\n", i + 1, theInfo.legendtexts[i]);
		bi_fprintf(bitFile, "xlegendfunction%d=\n", i + 1);
		bi_fprintf(bitFile, "ylegendfunction%d=\n", i + 1);
		bi_fprintf(bitFile, "fonttypelegendfunction%d=\n", i + 1);
		bi_fprintf(bitFile, "fontsizelegendfunction%d=\n\n", i + 1);
	}
	if (!standalone) {
		p = bi_getenv("BENCHIT_NODENAME", 1);
		char displayfileName[1000];
		sprintf(displayfileName, "%s/LOCALDEFS/%s_input_display", rootDir, p);
		freeCheckedC(&p);
		FILE *displayFile = fopen(displayfileName, "r");
		if (displayFile == 0) {
			sprintf(displayfileName, "%s/LOCALDEFS/PROTOTYPE_input_display", rootDir);
			displayFile = fopen(displayfileName, "r");
		}
		freeCheckedC(&rootDir);
		if (displayFile == 0) {
			printf(" [FAILED]\nBenchIT: Cannot open input file \"input_display\"\n");
			fclose(bitFile);
			return 1;
		}
		char fileReadBuf[1000];
		while (fgets(fileReadBuf, sizeof(fileReadBuf) - 1, displayFile) != (char *) 0) {
			bi_fprint(bitFile, fileReadBuf);
		}
		fclose(displayFile);
	} else {
		IDL(2, printf(" (run as standalone) "));
	}
	IDL(2, printf("...OK\nWriting Data...\n"));
	IDL(2, printf("Measurement count: %d; Values each: %d\n",theInfo.num_measurements, theInfo.numfunctions + 1));
	bi_fprintf(bitFile, "beginofdata\n");
	writeDataToFile(bitFile, results, theInfo.num_measurements, theInfo.numfunctions + 1);
	bi_fprintf(bitFile, "endofdata\n");
	IDL(2, printf("...OK\n"));
	fclose(bitFile);
	freeCheckedC(&kernelString);
	freeCheckedC(&language);
	for (i = 0; i < numLibraries; i++)
		freeCheckedC(&libraries[i]);
	freeCheckedC((char**) &libraries);
	return 0;
}

static int writeRawResults(char *bitFileName, bi_info theInfo, char* kernelName, double* results, int repeatCt) {
	if (theInfo.legendtexts == 0) {
		(void) fprintf(stderr, "BenchIT: info->legendtexts==NULL\n");
		(void) fflush(stderr);
		return 127;
	}

	char* rawFileName = bi_strndup(bitFileName, 4);
	strcat(rawFileName, ".raw");
	FILE *rawFile = fopen(rawFileName, "w");
	freeCheckedC(&rawFileName);
	if (rawFile == 0) {
		printf(" [FAILED]\nBenchIT: could not create output-file \"%s\"\n", rawFileName);
		return 127;
	}
	bi_fprintf(rawFile, "# BenchIT-RawFile\n#\n");
	bi_fprintf(rawFile, "#########################################################\n");

	IDL(2, printf("Writing definitions\n"));
	bi_fprintf(rawFile, "beginofdefinitions\n");
	bi_fprintf(rawFile, "bitfile=\"%s\"\n", bitFileName);
	bi_fprintf(rawFile, "kernelname=\"%s\"\n", kernelName);
	bi_fprintf(rawFile, "numfunctions=%d\n", theInfo.numfunctions);
	bi_fprintf(rawFile, "repeatct=%d\n\n", repeatCt);
	if (theInfo.xaxistext == 0)
		bi_fprintf(rawFile, "xaxistext=\n");
	else
		bi_fprintf(rawFile, "xaxistext=\"%s\"\n", theInfo.xaxistext);
	int i;
	for (i = 0; i < theInfo.numfunctions; i++) {
		if (theInfo.yaxistexts[i] == 0)
			bi_fprintf(rawFile, "y%daxistext=\n", i + 1);
		else
			bi_fprintf(rawFile, "y%daxistext=\"%s\"\n", i + 1, theInfo.yaxistexts[i]);
		if (theInfo.legendtexts[i] == 0)
			bi_fprintf(rawFile, "tlegendfunction%d=\n", i + 1);
		else
			bi_fprintf(rawFile, "tlegendfunction%d=\"%s\"\n", i + 1, theInfo.legendtexts[i]);
	}
	bi_fprintf(rawFile, "endofdefinitions\n");
	IDL(2, printf("Writing data\n"));
	bi_fprintf(rawFile, "beginofdata\n");
	writeDataToFile(rawFile, results, theInfo.num_measurements, theInfo.numfunctions * repeatCt + 1);
	bi_fprintf(rawFile, "endofdata\n");
	fclose(rawFile);
	return 0;
}

/*!****************************************************************************
 * Write *.bit-file (resultfile)
 * and *.bit.gp file (gnuplot file for quickview)
 */
int write_results(bi_info theInfo, double* results, int repeatCt, int standalone) {
	//First analyze results
	printf("BenchIT: Analyzing results...");
	fflush(stdout);
	int empty = 1, i;
	int dataPointCt = theInfo.numfunctions + repeatCt + 1;
	for (i = 0; i < theInfo.num_measurements; i++) {
		if (results[i * dataPointCt] >= 0) {
			empty = 0;
			break;
		} /* <=0: invalid */
	}
	if (empty) {
		printf("\nerror: No output data found, not writing result files\n");
		return 0;
	}
	double* bestResults = getBestResults(theInfo, results, repeatCt);
	if (bestResults == NULL ) {
		printf(" [FAILED]\nBenchIT: Could not get best results from data\n");
		return 1;
	}
	analyse_results(theInfo, bestResults);
	printf(" [OK]\n");

	printf("BenchIT: Writing resultfile...\n");
	fflush(stdout);
	IDL(2, printf("Writing measurementinfos\n"));
	/* composing result file name */
	char* str = (char*) malloc(sizeof(char) * 300);
	if (str == 0) {
		printf(" [FAILED]\nBenchIT: No more memory. ");
		freeCheckedD(&bestResults);
		return 1;
	}
	str[0] = 0;
	/* start with ARCH_SHORT */
	char* p = bi_getenv("BENCHIT_ARCH_SHORT", 0);
	if (p == 0 || standalone) {
		(void) strcat(str, "unknown");
	} else {
		(void) strcat(str, p);
		freeCheckedC(&p);
	}
	(void) strcat(str, "_");
	/* then ARCH_SPEED */
	p = (standalone) ? 0 : bi_getenv("BENCHIT_ARCH_SPEED", 0);
	if (p == 0) {
		(void) strcat(str, "unknown");
	} else {
		(void) strcat(str, p);
		freeCheckedC(&p);
	}
	(void) strcat(str, "__");

	/* The comment: BENCHIT_fileName_COMMENT */
	p = (standalone) ? 0 : bi_getenv("BENCHIT_fileName_COMMENT", 0);
	if (p == 0)
		(void) strcat(str, "0");
	else {
		(void) strcat(str, p);
		freeCheckedC(&p);
	}
	(void) strcat(str, "__");
	/* add date and time */
	time_t timeStamp = time((time_t *) 0);
	struct tm* curTime = localtime(&timeStamp);
	char timeBuf[35];
	(void) strftime(timeBuf, (size_t) sizeof(timeBuf), "%Y_%m_%d__%H_%M_%S.bit", curTime);
	(void) strcat(str, timeBuf);
	(void) strcat(str, "\0");
	char* fileName = (char*) malloc(sizeof(char) * strlen(str));
	(void) strcpy(fileName, str);
	freeCheckedC(&str);
	char* outputDir = (char*) calloc(300, sizeof(char));
	if (outputDir == 0) {
		printf(" [FAILED]\nBenchIT: No more memory. ");
		freeCheckedD(&bestResults);
		return 1;
	}
	outputDir[0] = 0;
	p = bi_getenv("BENCHIT_RUN_OUTPUT_DIR", 1); /* exit if default not set */
	IDL(5, printf("\noutput-dir=%s", p));
	(void) strcat(outputDir, p);
	(void) strcat(outputDir, "/");
	freeCheckedC(&p);
	char* kernelstring = bi_getenv("BENCHIT_KERNELNAME", 1);
	if (kernelstring == 0) {
		printf(" [FAILED]\nBenchIT: No kernelstring in info struct set. ");
		freeCheckedD(&bestResults);
		return 1;
	}
	/* replace all dots by / in p */
	for (i = 0; i <= length(kernelstring); i++)
		if (kernelstring[i] == '.')
			kernelstring[i] = '/';
	(void) strcat(outputDir, kernelstring);
	freeCheckedC(&kernelstring);
	if (chdir(outputDir) != 0) {
		if (createDirStructure(outputDir) != 0 || chdir(outputDir) != 0) {
			printf(" [FAILED]\nBenchIT: Couldn't change to output outputDir: %s.\n", outputDir);
			freeCheckedD(&bestResults);
			return 127;
		}
	}

	int res = writeBitFile(fileName, theInfo, bestResults, curTime, standalone);
	freeCheckedD(&bestResults);
	if (res != 0) {
		freeCheckedC(&outputDir);
		return res;
	}
	printf(" [OK]\nBenchIT: Wrote output to \"%s\" in directory\n", fileName);
	printf("         \"%s\"\n", outputDir);
	fflush(stdout);
	freeCheckedC(&outputDir);
	char* kernelName = bi_getenv("BENCHIT_KERNELNAME", 1);
	res = writeRawResults(fileName, theInfo, kernelName, results, repeatCt);
	if (res == 0)
		res = write_gp_file(fileName, kernelName, theInfo, xdata, ydata_global);
	freeCheckedC(&kernelName);
	return res;
}
