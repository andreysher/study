#ifdef __cplusplus
extern "C" {
#endif

#ifndef BENCH_IT_GNUWRITER_H
#define BENCH_IT_GNUWRITER_H

#include "../interface.h"

/*!@brief Holds information about an axis for assembling gnuplot options.
 *
 * See get_axis_properties().
 */
typedef struct axisdata {
	/*@{*/
	/*!@brief Minimum and maximum values from the measurement. */
	double min, max;
	/*@}*/
	/*@{*/
	/*!@brief Minimum and maximum values - gnuplot option. */
	double plotmin, plotmax;
	/*@}*/
	/*!@brief Base=0 for linear axis, 2 and 10 for logarithmic axis. */
	double base;
	/*!@brief Number of ticks on the axis - gnuplot option. */
	int ticks;
	/*!@brief The incrementation value - gnuplot option. */
	double incr;
	/*!@brief Name of the axis - 'x' or 'y'. */
	char name;
} axisdata;

int write_gp_file(char* bitFileName, char* kernelString, bi_info theInfo, axisdata xdata, axisdata ydata_global);
int get_axis_properties(axisdata *ad);

#endif

#ifdef __cplusplus
}
#endif
