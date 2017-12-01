#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <signal.h>
//количество итераций на тред после которого проверяем
#define ITERATIONS 1000

char flag = 1;
char catchFlag = 0;
pthread_barrier_t barrier;
int numberOfThreads;
double *partsOfSum;

void catchSignal(int signal){
	flag = 0;
}

void *threadFunction(void *p){
	int myNumber = (int)p;
	//printf("%d\n", myNumber);
	long long i;
	long long sumSteps = 0;
	long long lastEarly = myNumber;
	while(1){
		for(i = lastEarly; i < ((ITERATIONS * numberOfThreads) + lastEarly); i += numberOfThreads){
			partsOfSum[myNumber] += 1.0/(i*4.0 + 1.0);
			partsOfSum[myNumber] -= 1.0/(i*4.0 + 3.0);
		}
		lastEarly += i;
		sumSteps += i;
		pthread_barrier_wait(&barrier);
		if(flag == 0){
			catchFlag = 1;
		}
		pthread_barrier_wait(&barrier);
		if (catchFlag == 1)
		{
			break;
		}
	}
	pthread_exit(NULL);
}

int main(int argc, char const *argv[])
{
	if(argc != 2){
		printf("wrong number of arguments\n");
		exit(1);
	}

	numberOfThreads = atoi(argv[1]);
	if(numberOfThreads <= 0){
		printf("wrong number of threads\n");
		exit(1);
	}

	double pi = 0;
	pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * numberOfThreads);
	partsOfSum = (double*)calloc(numberOfThreads, sizeof(double));

	if((threads == NULL) || (partsOfSum == NULL)){
		printf("allocation problem\n");
		exit(1);
	}

	if(pthread_barrier_init(&barrier, NULL, numberOfThreads)){
		printf("problem with init barrier\n");
		exit(1);
	}

	signal(SIGINT, catchSignal);
    // signal(SIGTERM, catchSignal);
	int i;
	for (i = 0; i < numberOfThreads; ++i){
		if(pthread_create(threads + i, NULL, threadFunction, i) != 0){
			int j;
			for (j = 0; j < i; ++j)
			{
				pthread_join(threads[j], NULL);
			}
			printf("pthread create error\n");
			exit(1);
		}
	}

	for (i = 0; i < numberOfThreads; ++i)
	{
		pthread_join(threads[i], NULL);
		pi += partsOfSum[i];
	}

	pthread_barrier_destroy(&barrier);

	pi = pi*4;
	printf("pi = %.20f\n", pi);
	return 0;
}