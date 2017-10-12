#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "functions.h"
#define MORETHANKACH 200000000
#define MAXNOPS 500

int main(int argc, char const *argv[]){
	int q = 0;
	int i = 0;
	int j = 0;
	int k = 0;
	int tmp = 0;
	int time_ = 0;
	int *arr = calloc(MORETHANKACH, sizeof(int));
	FILE *rez = fopen("rezult.csv", "w");
	for (i = 0; i < MORETHANKACH; ++i)
	{
		arr[i] = i;
	}
	srand(time(NULL));
	for (i = MORETHANKACH - 1; i > 0; i--)
	{	
		
		k = i - 1;
		if (k != 0)
		{
			j = rand() % k;
		}
		tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;
	}
	puts("OK");
	for (i = 0; i < MAXNOPS; ++i)
	{	
		printf("%i\n", i);
		time_ = f[i](arr);
		printf("%i\n", time_);
		fprintf(rez, "%i,%i\n",i, time_);
	}
	return 0;
}

// http://www.ixbt.com/cpu/intel-haswell.shtml