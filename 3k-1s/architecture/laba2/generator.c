#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAXNOPS 500

int main(int argc, char const *argv[])
{	
	int i = 0;
	int k = 0;
	FILE *func = fopen("functions.h", "w");
	fprintf(func, "#include <time.h>\n\n");
	fprintf(func, "#include \"rdtsc.h\"\n\n");
	for(i = 0; i < MAXNOPS; i++){
		fprintf(func, "int func%i(int *arr){\n\
			int i = 2;\n\
			int j = 0;\n\
			unsigned long long start, end;\n\
			start = rdtsc();\n\
			for(j = 0; j < 1000; j ++){\n\
			i = arr[arr[i]];\n", i);
		for (k = 0; k < i; k++)
		{
			fprintf(func, "\t\t\tasm(\"nop\");\n");	
		}
		fprintf(func, "}\n");
		fprintf(func, "\t\t\tend = rdtsc();\n\
			return end - start;\n\
		}\n");
	}
	fprintf(func, "int(*f[%i])(int*) = {", MAXNOPS);
	for(k = 0; k < MAXNOPS; k++){
		if(k == (MAXNOPS - 1)){
			fprintf(func, "func%i", k);
		}
		else{	
			fprintf(func, "func%i,", k);
		}
	}
	fprintf(func, "};");
	return 0;
}