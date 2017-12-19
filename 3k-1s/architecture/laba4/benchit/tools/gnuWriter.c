#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "output.h"
#include "gnuWriter.h"

static const char *BIN_pref[] = { "", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei" }; /**< list of Prefixes*/
#define BIN_PREF_MAX 6 /**< number of Prefixes*/
static const char SI_prefixes[] = { 'y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', ' ', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y' }; /**< list of Prefixes defined by SI */
static const char *SI_pref = &(SI_prefixes[8]); /**< pointer to the SI_prefixes-array */
#define SI_PREF_MIN (-8) /**< min number of SI-Prefixes*/
#define SI_PREF_MAX 8 /**< min number of SI-Prefixes*/

/*!@brief Print contents of an axisdata struct for debugging purposes.
 * @param[in] ad Pointer to an axisdata structure.
 */
static void print_axisdata(const axisdata *ad) {
	printf("Min: %.20e\nMax: %.20e\nDiff: %.20e\nPlotmin: %e\nPlotmax: %e\n", ad->min, ad->max, ad->max - ad->min, ad->plotmin, ad->plotmax);
	printf("Base: %e\nTicks: %i\nIncr: %e\n", ad->base, ad->ticks, ad->incr);
	fflush(stdout);
}

/*!@brief Compute values for use by gnuplot.
 *
 * This function computes nice values for the generation of the gnuplot file.\n
 * It needs ad->min, ad->max and ad->base set.\n
 * The plotmin, plotmax, ticks and incr members of ad will be set.
 * @param[in,out] ad Pointer to an axisdata struct.
 */
int get_axis_properties(axisdata *axis) {
	if (DEBUGLEVEL >= 2)
		print_axisdata(axis);
	if (axis->base <= 1) { /* linear scale */
		double diff, tickspacing;
		/* minimal possible value for each axis is zero */
		axis->plotmin = 0.0;
		if(axis->max==BI_NEG_INFINITY || axis->min==BI_INFINITY){
			//invalid! (no data at all)
			axis->plotmax=0;
			axis->ticks=0;
			axis->incr=0;
			return 0;
		}
		/* difference between min and max */
		diff = axis->max - axis->min;
		if (diff <= 1e-30){
			//special case for min~max
			if(axis->min<=1e-30){
				//Only 1 value~0
				axis->ticks=1;
				axis->incr=1;
				axis->plotmax=axis->plotmin+axis->incr*axis->ticks;
				return 0;
			}
			diff=axis->min;
		}
		/* start with ticks like 1,2,3,4,5,6... */
		tickspacing = 1.0;
		/* if the difference is larger then 10 */
		if (diff > 10.0) {
			/* do ticks like 50,60,70,80 or 500,600,700,800 or according to difference */
			while (diff > 10.0) {
				diff /= 10.0;
				tickspacing *= 10.0;
			}
		} else {
			/* or ticks at .3,.4,.5... or less */
			while (diff < 1.0) {
				diff *= 10.0;
				tickspacing /= 10.0;
			}
		}

		if (diff <= 2) {
			/* use ticks at .1, .2, .3, ... */
			tickspacing /= 10.0;
		} else if (diff <= 2.5) {
			/* use ticks at .25, .5, .75 */
			tickspacing /= 4.0;
		} else if (diff <= 5.0) {
			/* use ticks at 0, 0.5, 1 */
			tickspacing /= 2.0;
		}
		/* set incr. */
		axis->incr = tickspacing;
		/* find the plot-min (should be a tick) */
		while (axis->plotmin + tickspacing < axis->min)
			axis->plotmin += tickspacing;
		/* set the max to min before finding max (incrementing, until its larger then the max measured value) */
		axis->plotmax = axis->plotmin;
		/* first, we have no ticks */
		axis->ticks = 0;
		/* find max */
		while (axis->plotmax <= axis->max) {
			axis->plotmax += tickspacing;
			axis->ticks += 1;
		}
	} else { /* logarithmic scale */
		/* be sure that it is displayable */
		if (axis->min <= 0.0) {
			(void) fprintf(stderr, "BenchIT: The minimum value of a result of your kernel is equal to or\n");
			(void) fprintf(stderr, "         smaller than 0.0, but logarithmic scaling was requested, which\n");
			(void) fprintf(stderr, "         is impossible for values <= 0.0\n");
			(void) fprintf(stderr, "         This is a bug in the kernel, contact its developer please.\n");
			(void) fflush(stderr);
			return 127;
		}
		/* find min ... */
		axis->plotmin = axis->base;
		/* for mins smaller then the base */
		if (axis->plotmin > axis->min) {
			while (axis->plotmin > axis->min)
				axis->plotmin /= axis->base;
		}
		/* or mins larger then the base */
		else {
			while (axis->plotmin <= (axis->min / axis->base))
				axis->plotmin *= axis->base;
		}
		/* set number of ticks and max's */
		axis->ticks = 0;
		axis->incr = axis->base;
		axis->plotmax = axis->plotmin;
		/* find max, which is a number like base^n and is larger then the real max */
		/* set the ticks accordingly */
		while (axis->plotmax < axis->max) {
			axis->plotmax *= axis->base;
			axis->ticks += 1;
		}
		/* remove all ticks, which are more then 10 */
		while (axis->ticks > 10) {
			if (axis->ticks & 0x1)
				axis->ticks++;
			axis->ticks /= 2;
			axis->incr *= axis->incr;
		}
	}
	if (DEBUGLEVEL >= 2)
		print_axisdata(axis);
	return 0;
}

/*!@brief Determine the proper prefix index and scaling for the given value.
 *
 * @param[in] value The number for which the scaling shall be computed.
 * @param[out] scaling_level Index for BIN_Pref or SI_PREF\n
 *             Positive for k, M, G, etc., negative for m, u, etc.
 * @param[out] scaling_value Multiply value with this number to get the proper
 *             scaled number for the prefix.
 * @param[in] bBase2Scaling Boolean value to select the use of the scaling based
 *            on 1024 (@c bBase2Scaling<>0) rather than 1000 (@c bBase2Scaling=0).
 */
static void get_scaling(double value, int * scaling_level, double * scaling_value, int bBase2Scaling) {
	/* decimal scaling */
	double scaling_base = 1000.0;
	/* binary scaling */
	if (bBase2Scaling)
		scaling_base = 1024.0;
	/* first no scale active */
	*scaling_level = 0;
	*scaling_value = 1.0;
	/* no scaling wanted */
	if ((value >= 0) && (value <= 0))
		return;
	/* <0? get absolute value */
	if (value < 0)
		value = fabs(value);
	/* find best scaling level by going through data */
	if (value >= scaling_base) {
		while (value >= scaling_base) {
			value = (double) (value / scaling_base);
			*scaling_level += 1;
		}
	} else if ((value < 1.0) && (bBase2Scaling == 0)) { /* "negative" scaling not possible for base 2 */
		while (value < 1.0) {
			value = (double) (value * scaling_base);
			*scaling_level -= 1;
		}
	}
	/* set to max or min possibble scaling */
	if (bBase2Scaling) {
		if (*scaling_level > BIN_PREF_MAX)
			*scaling_level = BIN_PREF_MAX;
	} else {
		if (*scaling_level > SI_PREF_MAX)
			*scaling_level = SI_PREF_MAX;
		else if (*scaling_level < SI_PREF_MIN)
			*scaling_level = SI_PREF_MIN;
	}
	/* scale it */
	*scaling_value = pow(scaling_base, (*scaling_level) * (-1));
}

/*!@brief Generate settings for gnuplot from an axisdata struct and write to
 *        an already opened file.
 *
 * This function generates the settings for a pretty output from gnuplot.\n
 * It applies size prefixes for the values on the axes and makes ticks on
 * 'nice' positions.\n
 * The settings will be appended to the file denoted by f.
 * @param[in] f An open FILE *with permission to write.
 * @param[in] ad An axisdata struct which was already processed by
 *            get_axis_properties()
 */
static void write_gnuplot_axisproperties(FILE *file, const axisdata *ad) {
	double pos = 0.0, pos1 = 0.0, scaling_value = 0.0;
	int prefIndex = 0, i = 0, first = 1;

	switch ((int) ad->base) {
	case 0: { /* linear scale */
		fprintf(file, "set %ctics (", ad->name);
		/* search for value nearest to 0 for determining the scaling */
		double minVal = 1e30;
		for (pos = ad->plotmin; pos <= ad->plotmax; pos += ad->incr) {
			if (fabs(pos) < minVal && pos != 0.0)
				minVal = pos;
		}
		get_scaling(minVal, &prefIndex, &scaling_value, 0);
		for (pos = ad->plotmin; pos <= ad->plotmax; pos += ad->incr) {
			pos1 = pos * scaling_value;
			if (first) {
				fprintf(file, "\"%.6g%c\" %e", pos1, SI_pref[prefIndex], pos);
				first = 0;
			} else {
				fprintf(file, ",\"%.6g%c\" %e", pos1, SI_pref[prefIndex], pos);
			}
		}
		fprintf(file, ")\n");
		break;
	}
	case 2: {
		fprintf(file, "set %ctics (", ad->name);
		/* search for value nearest to 0 for determining the scaling */
		double minVal = 1e30;
		for (pos = ad->plotmin; pos <= ad->plotmax; pos += ad->incr) {
			if (fabs(pos) < minVal && pos != 0.0)
				minVal = pos;
		}
		get_scaling(minVal, &prefIndex, &scaling_value, 0);
		for (pos = ad->plotmin; pos <= ad->plotmax; pos *= ad->incr) {
			pos1 = pos * scaling_value;
			if (first) {
				fprintf(file, "\"%.6g%s\" %e", pos1, BIN_pref[prefIndex], pos);
				first = 0;
			} else
				fprintf(file, ",\"%.6g%s\" %e", pos1, BIN_pref[prefIndex], pos);
		}
		fprintf(file, ")\n");
		fprintf(file, "set logscale %c %g\n", ad->name, ad->base);
		break;
	}
	case 10: {
		fprintf(file, "set %ctics (", ad->name);
		for (pos = ad->plotmin; pos <= ad->plotmax; pos *= ad->incr) {
			get_scaling(pos, &prefIndex, &scaling_value, 0);
			pos1 = pos * scaling_value;
			if (first) {
				fprintf(file, "\"%.6g%c\" %e", pos1, SI_pref[prefIndex], pos);
				first = 0;
			} else {
				fprintf(file, ",\"%.6g%c\" %e", pos1, SI_pref[prefIndex], pos);
			}
			if (ad->ticks <= 5)
				for (i = 2; i <= 9; ++i)
					fprintf(file, ",\"\" %e 1", pos * i);
		}
		fprintf(file, ")\n");
		fprintf(file, "set logscale %c %g\n", ad->name, ad->base);
		break;
	}
	default:
		fprintf(file, "set %ctics %e,%e\n", ad->name, ad->plotmin, ad->incr);
		break;
	}
	fprintf(file, "set %crange [%e:%e]\n", ad->name, ad->plotmin, ad->plotmax);
}

int write_gp_file(char* bitFileName, char* kernelString, bi_info theInfo, axisdata xdata, axisdata ydata_global) {
	printf("BenchIT: Writing quickview file...\n");
	fflush(stdout);
	char *gpFileName=bi_strndup(bitFileName,3);
	strcat(gpFileName, ".gp");
	FILE *gpFile = fopen(gpFileName, "w");
	freeCheckedC(&gpFileName);
	fprintf(gpFile, "#gnuplotfile\n");
	fprintf(gpFile, "set title \"%s\"\n", kernelString != 0 ? kernelString : "");
	fprintf(gpFile, "set xlabel \"%s\"\n", theInfo.xaxistext != 0 ? theInfo.xaxistext : "");
	if ((xdata.ticks != 0)) {
		write_gnuplot_axisproperties(gpFile, &xdata);
	}
	if (get_axis_properties(&ydata_global))
		return 127;
	if ((ydata_global.ticks != 0)) {
		write_gnuplot_axisproperties(gpFile, &ydata_global);
	}
	fprintf(gpFile, "set ylabel \"%s\"\n", (theInfo.yaxistexts != 0) && (theInfo.yaxistexts[0] != 0) ? theInfo.yaxistexts[0] : "");
	if (bi_getenv("BENCHIT_LINES", 0))
		fprintf(gpFile, "set data style linespoints\n");
	else
		fprintf(gpFile, "set data style points\n");
	fprintf(gpFile, "set term postscript eps color solid\n");
	char *epsFileName=bi_strndup(bitFileName,4);
	strcat(epsFileName, ".eps");
	fprintf(gpFile, "set output \"%s\"\n", epsFileName);
	freeCheckedC(&epsFileName);
	if (theInfo.gnuplot_options != NULL )
		fprintf(gpFile, "%s\n", theInfo.gnuplot_options);
	int i;
	for (i = 0; i < theInfo.numfunctions; i++) {
		if (i != 0)
			fprintf(gpFile, ",");
		else
			fprintf(gpFile, "plot");
		fprintf(gpFile, " \"%s\" using 1:%d title '%s'", bitFileName, i + 2, theInfo.legendtexts[i] != 0 ? theInfo.legendtexts[i] : "");
	}
	fclose(gpFile);
	printf(" [OK]\n");
	fflush(stdout);
	return 0;
}
