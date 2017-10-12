#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

void *start_routine(void * param) {
	while(1){
		printf("string of text\n");
	//	sleep(1);
	}
}

int main(int argc, char *argv[]) {
    pthread_t child;
    if(!pthread_create(&child, NULL, start_routine, NULL)){
    	//sleep(2);
    	if(pthread_cancel(child)){
    		printf("incorrect pthread_cancel\n");
    	}
    }
    else{
    	printf("incorrect create thread\n");
    }
    return 0;
}
