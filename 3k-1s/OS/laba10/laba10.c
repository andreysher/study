#include<stdio.h>
#include<pthread.h>
#include<unistd.h>

#define ITERS 10
#define MUTEX_QUANTITY 3
pthread_mutex_t *mutArr;
int childStarted = 0;

void printed(int startUnlock, int startLock, char* str){
    int i;
    int willUnlock = startUnlock;
    int willLock = startLock;
    for(i = 0; i < ITERS; i++){
        printf("From %s string %d\n", str, i);
        pthread_mutex_unlock(mutArr + willUnlock);
        pthread_mutex_lock(mutArr + willLock);
        willUnlock = (willUnlock+1)%MUTEX_QUANTITY;
        willLock = (willLock+1)%MUTEX_QUANTITY;
    }
    pthread_mutex_unlock(mutArr + (willUnlock)%MUTEX_QUANTITY);
    pthread_mutex_unlock((mutArr + (willLock - 1)%MUTEX_QUANTITY));
}

void *childFunc(void* param){
    pthread_mutex_lock(mutArr + 1);
    childStarted = 1;
    pthread_mutex_lock(mutArr + 2);

    printed(1, 0, "child");

    pthread_exit(NULL);
}

int main(int argc, char const *argv[]){   

    mutArr = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t) * MUTEX_QUANTITY);

    pthread_mutexattr_t atr;

    pthread_mutexattr_init(&atr);
    pthread_mutexattr_settype(&atr, PTHREAD_MUTEX_ERRORCHECK);

    int i;
    for(i = 0; i<MUTEX_QUANTITY; i++){
        pthread_mutex_init(mutArr + i, &atr);
    }

    pthread_t thread;

    pthread_mutex_lock(mutArr);
    pthread_mutex_lock(mutArr + 2);

    pthread_create(&thread, NULL, (void*)&childFunc, NULL);

    while(!childStarted){sched_yield();}

    printed(2, 1, "parent");

    pthread_join(thread, NULL);

    for(i = 0; i < MUTEX_QUANTITY; i++){
        pthread_mutex_destroy(mutArr + i);
    }

    pthread_exit(NULL);
}