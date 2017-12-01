#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>

#define PHILO 5
#define DELAY 30000
#define FOOD 50

pthread_mutex_t foodLock;
pthread_mutex_t forkLock;

pthread_mutex_t forks[PHILO];
pthread_t philosophers[PHILO];

pthread_cond_t forkCond;

int sleep_seconds = 0;

struct parameter
{
	int id;
};

void get_fork(int phil, int fork, const char hand[])
{
	pthread_mutex_lock(&forks[fork]);
	
	printf("Philosopher %d: got %s fork %d\n", phil, hand, fork);
}

void get_forks(int phil, int fork1, int fork2)
{
	pthread_mutex_lock(&forkLock);

	int res1;
	int res2;

	printf("up   %d %d\n", fork1, fork2);

	bool busy;

	do
	{
		busy = false;

		res1 = pthread_mutex_trylock(&forks[fork1]);

		if (0 == res1)
		{
			res2 = pthread_mutex_trylock(&forks[fork2]);

			if (EBUSY == res2)
			{
				pthread_mutex_unlock(&forks[fork1]);
			
				busy = true;
			}
			else if (0 != res2)
			{
				//perror("Error while second lock.\n");
			}
		}
		else if (EBUSY == res1)
		{
			busy = true;
		}
		else
		{
			//error
		}

		if (busy)
		{
			pthread_cond_wait(&forkCond, &forkLock);
		}
	}
	while (busy);

	printf("done\n");
	
	pthread_mutex_unlock(&forkLock);
}

void down_forks(int f1, int f2)
{
	pthread_mutex_lock(&forkLock);

	pthread_mutex_unlock(&forks[f1]);
	pthread_mutex_unlock(&forks[f2]);

	printf("down %d %d\n", f1, f2);

	pthread_cond_broadcast(&forkCond);

	pthread_mutex_unlock(&forkLock);
}

int food_on_table()
{
	static int food = FOOD;

	printf("%d\n", food);
	
	pthread_mutex_lock(&foodLock);
	
	if(food > 0) 
	{
		food--;
	}
	
	int myfood = food;
	pthread_mutex_unlock(&foodLock);
	
	return myfood;
}

void *philosopher(void *parameter)
{
	int f;

	int id =(*(struct parameter *)parameter).id;

	printf("Philosopher %d sitting down to dinner.\n", id);

	int left_fork = id % PHILO;
	int right_fork = (id + 1) % PHILO;

	while(0 != (f = food_on_table())) 
	{
		if(id == 1)
		{
			sleep(sleep_seconds);
		}

		printf("Philosopher %d: get dish %d.\n", id, f);
		
/*		get_fork(id, right_fork, "right");
		get_fork(id, left_fork, "left ");
*/
		get_forks(id, right_fork, left_fork);

		printf("Philosopher %d: eating.\n", id);
		
		usleep(DELAY * (FOOD - f + 1));
		down_forks(left_fork, right_fork);
	}
	
	printf("Philosopher %d is done eating.\n", id);
	
	return NULL;
}

int main(int argc, char **argv)
{
	if(argc == 2)
	{
		sleep_seconds = atoi(argv[1]);
	}

	struct parameter *parameters = new struct parameter[PHILO];

	pthread_mutex_init(&foodLock, NULL);
	pthread_mutex_init(&forkLock, NULL);

	pthread_cond_init(&forkCond, NULL);
	
	for(int i = 0; i < PHILO; i++)
	{
		pthread_mutex_init(&forks[i], NULL);
	}

	for(int i = 0; i < PHILO; i++)
	{
		parameters[i].id = i;

		pthread_create(&philosophers[i], NULL, philosopher, &parameters[i]);
	}

	for(int i = 0; i < PHILO; i++)
	{
		pthread_join(philosophers[i], NULL);
	}
	
	for(int i = 0; i < PHILO; i++)
	{
		pthread_mutex_destroy(&forks[i]);
	}

	pthread_cond_destroy(&forkCond);

	pthread_mutex_destroy(&forkLock);
	pthread_mutex_destroy(&foodLock);

	delete[] parameters;

	return 0;
}