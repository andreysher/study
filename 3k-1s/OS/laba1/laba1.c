#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

void *start_routine(void * param) {
    int i; 
    for(i = 0; i < 10; i++){
        // write(stdin,param,5);
        printf("child's string of text %i\n", i);
        // sleep(5);
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
    char *str = "text";
    code=pthread_create(&thread, NULL, start_routine, str);
    if (code==0) {
        for(i = 0; i < 10; i++){
            // write(stdin, str, 5);
            printf("parent's string of text %i\n", i);
            // sleep(5);
        }
    }
    else{
        printf("incorrect created thread\n");
    }
    // sleep(10);
    pthread_exit(0);
    // return 0;
}