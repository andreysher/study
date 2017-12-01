#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>

#define PHILO 10
#define DELAY 300000
#define FOOD 40

pthread_mutex_t foodLock, forkLock;
pthread_cond_t forkCond;
pthread_mutex_t forks[PHILO];
pthread_t philosophers[PHILO];

typedef struct parameter{
	int id;
} param;

void get_forks(int phil, int fork1, int fork2){
	pthread_mutex_lock(&forkLock);
	printf("locking forks %d\n", phil);
	int res1, res2;
	int busy = 0;

	do{
		if(busy){
			printf("sleep %d\n", phil);
			pthread_cond_wait(&forkCond,&forkLock);
			printf("wake up %d\n", phil);
		}
		busy = 0;
		res1 = pthread_mutex_trylock(&forks[fork1]);
		if(0 == res1){
			res2 = pthread_mutex_trylock(&forks[fork2]);
			if(EBUSY == res2){
				pthread_mutex_unlock(&forks[fork1]);
				busy = 1;
			}
		}
		else if(EBUSY == res1){
			busy = 1;
		}
		else{
			printf("error in trylock\n");
		}
	}while(busy);

	pthread_mutex_unlock(&forkLock);
	printf("unlocking forks %d\n", phil);
}

void down_forks(int f1, int f2){
	
	pthread_mutex_unlock(&forks[f2]);
	pthread_mutex_unlock(&forks[f1]);

	pthread_mutex_lock(&forkLock);
	pthread_cond_broadcast(&forkCond);

	pthread_mutex_unlock(&forkLock);
}

int food_on_table(){
	static int food = FOOD;
	pthread_mutex_lock(&foodLock);
	if(food > 0){
		food--;
	}
	int my = food;
	pthread_mutex_unlock(&foodLock);

	return my;
}

void *philosopher(void* paramet){
	int id = ((param*)paramet)->id;

	int left_fork = id % PHILO;
	int right_fork = (id + 1) % PHILO;

	int f;
	int eaten = 0;
	while(0 != (f = food_on_table())){
		get_forks(id, right_fork, left_fork);
		
		printf("Philosopher %d: eating.\n", id);
		
		usleep(DELAY * (FOOD - f + 1));
		down_forks(left_fork, right_fork);
		
		printf("down %d\n", id);
		eaten++;
	}
	printf("philosopher %d eat %d dishes\n", id, eaten);
}

int main(int argc, char const *argv[])
{
	pthread_mutex_init(&foodLock, NULL);
	pthread_mutex_init(&forkLock, NULL);
	pthread_cond_init(&forkCond, NULL);

	param parametr[PHILO];

	int i, j;
	int error;
	for (i = 0; i < PHILO; ++i)
	{
		error = pthread_mutex_init(&forks[i], NULL);
		if(error){
			printf("error in mutex init\n");
			for (j = 0; j < i; ++j)
			{
				error = pthread_mutex_destroy(&forks[j]);
				if(error){
					printf("error in mutex destroy\n");
					exit(1);
				}
			}
			exit(1);
		}
	}

	for (i = 0; i < PHILO; ++i){
		parametr[i].id = i;
		error = pthread_create(&philosophers[i], NULL, philosopher, &parametr[i]);
		if(error){
			printf("error in threads creating\n");
			for(j = 0; j < i; ++j){
				pthread_join(philosophers[j], NULL);
			}
			exit(1);
		}
	}

	for (i = 0; i < PHILO; ++i)
	{
		error = pthread_join(philosophers[i], NULL);
		if(error){
			printf("error in joining\n");
			exit(1);
		}
	}

	for (i = 0; i < PHILO; ++i)
	{
		error = pthread_mutex_destroy(&forks[i]);
		if (error)
		{
			printf("error in destroy mutex\n");
			exit(1); 
		}
	}

	pthread_cond_destroy(&forkCond);

	pthread_mutex_destroy(&foodLock);
	pthread_mutex_destroy(&forkLock);

	return 0;
}