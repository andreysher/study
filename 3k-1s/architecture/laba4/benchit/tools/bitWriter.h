#ifdef __cplusplus
extern "C" {
#endif

#ifndef BENCH_IT_BITWRITER_H
#define BENCH_IT_BITWRITER_H

#include "../interface.h"

#define MAX_ADD_INFO_STR 256

int initResults(bi_info theInfo);

/**
* @Brief Writes the results to .bit and .bit.gp file
*
* ONLY call from MPI rank 0 as output is given without further checks
*
* @param(in) theInfo Kernel information structure
* @param(in) results Result array with each entry = (x y1+ y2+...)
* @param(in) funcCt Number of functions (yn)
* @param(in) repeatCt Number of measurements
* @returns 0 for OK or anything else for error
*/
int write_results(bi_info theInfo, double* results, int repeatCt, int standalone);

#endif

#ifdef __cplusplus
}
#endif
