#include <stdlib.h>
#include <time.h>
#include "rdtsc.h"
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>

#define MEASSURES 20

int test(int LSpy)
{	
	int l = 0;
    int long unsigned i;
    int a = 1;
    int long unsigned liter = 10000000;
    for (i = 0; i < liter; ++i)
    {	if(i%2) l++; //spy branch
    	if(i%2) l++;
    	if(i%2) l++;
    	if(i%2) l++;
        if ((i % LSpy) == 0) a++; 
    }
    return a;
}

void meassure(int LSpy)
{
    float average = 0;
    uint64_t min = -1;
    uint64_t max = 0;
    int i;
    for (i = 0; i < MEASSURES; i++){
        unsigned long long start, finish;

        start = rdtsc();
        test(LSpy);
        finish = rdtsc();

        unsigned long long current = finish - start;
        average += (float) current;
        if (current < min) min = current;
        if (current > max) max = current;
    }

    average /= MEASSURES;
    printf("%d, %lu\n", LSpy, min);
}

int main(void)
{
    for (int i = 1; i < 100; ++i)
    {
        meassure(i);
    }
}

//https://docs.google.com/spreadsheets/d/1TNQvMN_ZxEZVl29W9abkRZssa1Q7eR14Jq1EVVO0HPw/edit#gid=1552015687
//https://docs.google.com/spreadsheets/d/1aoJHgSLJlx_reIjQlD7bF5OZTAdM7BtJ4J0HCAyFpSE/edit#gid=111120519
//https://docs.google.com/spreadsheets/d/1YduoOSo06sgX7Sd9trkL3nX3troMkn7Bn4iHiUNitNs/edit#gid=839642930