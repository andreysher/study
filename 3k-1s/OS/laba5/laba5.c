#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>



void svertka(void * param){

    printf("menya ubili\n");
}

void *start_routine(void * param) {
    pthread_cleanup_push(svertka, NULL);
    while(1){
        printf("string of text\n");
        sleep(1);
    }
    pthread_cleanup_pop(1);
}

int main(int argc, char *argv[]) {
    pthread_t child;
    if(!pthread_create(&child, NULL, start_routine, NULL)){
        
    	sleep(2);
    	if(pthread_cancel(child)){
    		printf("incorrect pthread_cancel\n");
    	}
    }
    else{
    	printf("incorrect create thread\n");
    }
    pthread_exit(0);
}
