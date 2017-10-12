#include <pthread.h>
#include <stdio.h>
#include <string.h>

pthread_mutexattr_t attr;
pthread_mutex_t m1=PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  cond=PTHREAD_COND_INITIALIZER;

void* print_message ( void* str ){
    int i = 0;

    pthread_mutex_lock( &m1 );  
    for ( i = 0; i < 10; i++ ){
        pthread_cond_signal ( &cond );
        printf ( "Message : %s\n",(char*) str );
        pthread_cond_wait ( &cond, &m1);
    }
    pthread_mutex_unlock( &m1 );
    pthread_cond_signal ( &cond );
    return NULL;
}

int main ( int argc, char* argv ){
    pthread_t thread;
    int err;

    if ( err= pthread_create (&thread, NULL, print_message, (void*)"child" )){
        fprintf ( stderr, "Error in creating thread %s\n", strerror(err));
    }

    print_message ( (void*) "parent" );

    pthread_join(thread, NULL);
    pthread_mutex_destroy( &m1 );
    return 0;
}
