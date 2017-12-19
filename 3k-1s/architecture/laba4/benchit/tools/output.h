#ifdef __cplusplus
extern "C" {
#endif

#ifndef BENCH_IT_OUTPUT_H
#define BENCH_IT_OUTPUT_H

//For FILE reference
#include <stdio.h>
/*!@brief Create the directories of the pathname denoted by dirstr.
 *
 * If any part of the pathname is not accessible or can not be createad, the
 * function will return non zero value
 * @param dirstr The pathname which shall be created.
 */
/* creates directory structure or exits if that's not possible */
int createDirStructure(const char *dirstr);
void bi_fprint(FILE *f, char *s);
void bi_fprintf(FILE *f, char *s, ...);
/* these values are used for getting max and mins. They will be setted when starting*/
extern double BI_INFINITY; /**< default values to be used if values are to big, set when starting main()*/
extern double BI_NEG_INFINITY; /**< default values to be used if values are to small, set when starting main()*/
#endif

#ifdef __cplusplus
}
#endif
