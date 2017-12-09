#include <pthread.h>
#include <stdio.h>
#include <string.h>
#define ITERATIONS 10

pthread_mutexattr_t attr;
pthread_mutex_t m1=PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  cond=PTHREAD_COND_INITIALIZER;

typedef struct par{
    int *currentFlag;
    int *otherFlag;
    void* name;
} params;

void* print_message ( void* parames ){
    int i = 0;
    params *parametrs = (params*) parames;
    for ( i = 0; i < ITERATIONS; i++ ){
        pthread_mutex_lock( &m1 );
        while(!*(parametrs->currentFlag)){
            pthread_cond_wait(&cond, &m1);
        }

        printf ( "Message : %s\n",(char*) parametrs->name );
        *(parametrs->currentFlag) = 0;
        *(parametrs->otherFlag) = 1;

        pthread_cond_signal(&cond);

        pthread_mutex_unlock(&m1);
    }
    return NULL;
}


int main ( int argc, char* argv ){
    pthread_t thread;
    int err;

    params parentPar;
    params childPar;

    int childFlag = 0;
    int parentFlag = 1;

    parentPar.currentFlag = &parentFlag;
    parentPar.otherFlag = &childFlag;
    parentPar.name = "parent";

    childPar.currentFlag = &childFlag;
    childPar.otherFlag = &parentFlag;
    childPar.name = "child";

    if ( err= pthread_create (&thread, NULL, print_message, (void*)(&childPar) )){
        fprintf ( stderr, "Error in creating thread %s\n", strerror(err));
    }
    print_message ( (void*)(&parentPar) );

    pthread_join(thread, NULL);
    pthread_cond_destroy(&cond);
    pthread_mutex_destroy( &m1 );
    return 0;
}
