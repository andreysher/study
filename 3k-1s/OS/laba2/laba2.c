#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

void *start_routine(void * param) {
    int i; 
    for(i = 0; i < 10; i++){
        printf("child's string of text %i\n", i);
        sleep(1);
    }
}

/*
 * creates thread with default attributes and no parameters
 * sleeps enough time to ensure that thread will print his message
 * (under reasonable conditions)
 */
int main(int argc, char *argv[]) {
    pthread_t thread;
    int code;
    int i;
    
    code=pthread_create(&thread, NULL, start_routine, NULL);
    pthread_join(thread,NULL);
    if (code==0) {
        for(i = 0; i < 10; i++){
            printf("parent's string with text %i\n", i);
            sleep(1);
        }
    }
    else{
        printf("incorrect created thread\n");
    }
    pthread_exit(1);
    return 0;
}