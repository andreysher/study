#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#define num_steps 2000000

union params{
	long long threadNumber;
	double lockalPi;
};

unsigned long threadsQuantity;

void *start_routine(void* param) {

	double pi = 0;
    long long i = ((union params *)param)->threadNumber;
    if(i > num_steps){
        return;
    }
    for (i; i < num_steps; i += threadsQuantity) {
         pi += 1.0/(i*4.0 + 1.0);
         pi -= 1.0/(i*4.0 + 3.0);
    }
    
    ((union params *)param)->lockalPi = pi;

    return param;

}

int main(int argc, char** argv) {
	int i;
    pthread_t* threads;
 	union params * param;
 	double PI = 0;

 	threadsQuantity = atol(argv[1]);
    printf("%lu\n", threadsQuantity);
    if(threadsQuantity >= sizeof(size_t)){
        return;
    }
 	param = malloc(threadsQuantity * sizeof(union params));
 	threads = malloc(threadsQuantity * sizeof(pthread_t));
 	for (i = 0; i < threadsQuantity; ++i){
 		param[i].threadNumber = i;
 		pthread_create(&(threads[i]), NULL, start_routine, (void*)&(param[i]));
 	}
 	for (i = 0; i < threadsQuantity; ++i){
 		union params * res;
 		pthread_join(threads[i], (void **)&res);
 		PI += res->lockalPi;
 	}
 	PI *= 4.0;
 	printf("pi done - %.16g \n", PI);

 	return 0;
}