#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
/* main posix header */
#include <unistd.h>
#include <stdarg.h>

#include "stringlib.h"

/*!@brief Create the directories of the pathname denoted by dirstr.
 *
 * If any part of the pathname is not accessible or can not be createad, the
 * function will return non zero value
 * @param dirstr The pathname which shall be created.
 */
/* creates directory structure or exits if that's not possible */
int createDirStructure(const char *dirstr) {
	/* buffer for string work */
	char buf[100];
	int flag = 0, i = 0, len = length(dirstr) + 1, pos = -1, last = 0;
	int successFlag = 1;
	/* set the content of the buffer to 000000... */
	memset(buf, 0, 100);
	/* if the directory starts with an / (is the root-directory) */
	if (dirstr[0] == '/') {
		/* start at / and go through all subdirs one by one */
		while (last >= 0) {
			/* char-position after the actual subfolders name */
			i = indexOf(dirstr, '/', last);
			pos = i;
			/* if there is no ending / */
			if (pos < 0)
				pos = len - 1;
			/* get actual subfolders name and write it to buf */
			substring(dirstr, buf, last, pos + 1);
			/* try to change into the subdir */
			flag = chdir(buf);
			if (flag != 0) {
				/* if it doesn't exist, try to create it */
				mkdir(buf, S_IRWXU);
				/* if it couldnt be created */
				if (chdir(buf) != 0) {
					successFlag = 0;
				}
			}
			/* no more '/'s found */
			if (i < 0)
				last = -1;
			/* more found, go to the next subfolder */
			else
				last = pos + 1;
		}
	}
	/* couldn't create subfolder? */
	if (successFlag != 1) {
		printf(" [FAILED]\nBenchIT: could not create directory structure \"%s\"\n", dirstr);
		return 127;
	}
	return 0;
}

/*!@brief prints a string to File
 * @param[in] f a FILE* to write a string to
 * @param[in] s a char* to write into f
 */
void bi_fprint(FILE *f, char *s) {
	int a = 0, i = 0, j = 0, k = 0;
	char buf[82];
	/* sth doesnt exist? */
	if (f == 0)
		return;
	if (s == 0)
		return;
	/* get length */
	a = (int) strlen(s);
	/* length <=79: write line */
	if (a <= 79)
		fprintf(f, "%s", s);
	/* else line shall be broken */
	else {
		for (i = 0; i < a; i++) {
			if (s[i] == '\t')
				s[i] = ' ';
			if ((s[i] == '\\') || (s[i] == '\n')) { /* manual line break */
				buf[j] = s[i];
				if (s[i] == '\n')
					buf[j + 1] = '\0';
				else {
					buf[j + 1] = '\n';
					buf[j + 2] = '\0';
				}
				fprintf(f, "%s", buf);
				j = 0;
				k = 0;
				continue;
			}
			if (s[i] == ' ') /* remember last space */
				k = j;
			if (j == 79) { /* eol reached */
				if (k == 0)
					k = j;
				buf[k] = '\\';
				buf[k + 1] = '\n';
				buf[k + 2] = '\0';
				fprintf(f, "%s", buf);
				i = i - (j - k);
				j = 0;
				k = 0;
				continue;
			}
			buf[j] = s[i]; /* copy char */
			j++;
		}
	}
}

void bi_fprintf(FILE *f, char *s, ...) {
	char buf[100000];
	va_list args;
	va_start(args, s);
	vsprintf(buf, s, args);
	bi_fprint(f, buf);
	va_end(args);
}
