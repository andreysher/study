#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define COUNT_OF_THREADS 4
#define LEN_OF_STRINGS 10

void *print_strings(void *param){
	char *par = (char *)param;
	int i = atoi(param);
	while(i){
		printf("%s\n", param);
		i--;
	}
}

int main(){
	pthread_t threads[COUNT_OF_THREADS];
	char threads_params[COUNT_OF_THREADS][LEN_OF_STRINGS] = {"3","2","1","4"};
	int i, code;
	for (i = 0; i < COUNT_OF_THREADS; ++i)
	{
		code = pthread_create(&(threads[i]), NULL, print_strings, &(threads_params[i]));
	}
	for(i = 0; i < COUNT_OF_THREADS; ++i){
		code = pthread_join(threads[i],NULL);
	}
	return 0;
}