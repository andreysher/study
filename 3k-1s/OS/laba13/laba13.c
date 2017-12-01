#include<stdio.h>
#include<pthread.h>
#include<semaphore.h>
#include<string.h>

#define ITER 10

typedef struct parameter
{
	sem_t *currentSem;
	sem_t *otherSem;
	char *name;
} param;

void *printed(param * par){
	int i = 0;
	for (i = 0; i < ITER; i++)
	{
		sem_wait(par->otherSem);

		printf("Line %d from thread %s\n", i, par->name);
		sem_post(par->currentSem);
	}
}

int main(int argc, char const *argv[])
{
	pthread_t thread;

	sem_t child_semaphor;
	sem_t parent_semaphor;

	sem_init(&child_semaphor, 0, 1);
	sem_init(&parent_semaphor, 0, 0);

	param childParam = {&child_semaphor, &parent_semaphor, "child"};

	if(0 != pthread_create(&thread, NULL, (void*)&printed, (void*)&childParam)){
		perror("error while thread create");
	}
	else{
		param parentParam = {&parent_semaphor, &child_semaphor, "parent"};
		printed(&parentParam);

		if(0 != pthread_join(thread, NULL)){
			perror("error while join");
		}
	}

	sem_destroy(&child_semaphor);
	sem_destroy(&parent_semaphor);

	return 0;
}