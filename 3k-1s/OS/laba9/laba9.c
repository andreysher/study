#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
//запустить на другой системе + рассказать про usleep
#define PHILO 5
#define DELAY 1000
#define FOOD 50
#define BUSY 1
#define RIGHT 0
#define LEFT 1

pthread_mutex_t forks[PHILO];
pthread_t phils[PHILO];
pthread_mutex_t food_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t forks_mut = PTHREAD_MUTEX_INITIALIZER;

void down_forks(int left_fork, int rigth_fork){
	pthread_mutex_unlock(&forks[left_fork]);
	pthread_mutex_unlock(&forks[rigth_fork]);
}

int get_fork(int phil_ID, int fork_ID, char hand){
	pthread_mutex_lock(&forks[fork_ID]);
}

int get_forks(int phil_ID,int rigth_fork_ID){
	pthread_mutex_lock(&forks_mut);
	get_fork(phil_ID, rigth_fork_ID, RIGHT);
	int left_fork_ID = rigth_fork_ID+1;
	if(left_fork_ID == PHILO)
		left_fork_ID = 0;
	get_fork(phil_ID, left_fork_ID, LEFT);
	pthread_mutex_unlock(&forks_mut);
}



int food_on_table(){
	static food = FOOD;
	int myfood;
	pthread_mutex_lock(&food_lock);
	if(food > 0){
		food--;
	}
	myfood = food;
	pthread_mutex_unlock(&food_lock);
	return myfood;
}

void *philosopher(void* num){
	int f;
	int rigth_fork = (int)num;
	int left_fork = rigth_fork + 1;
	int counter = 0;
	if(left_fork == PHILO)
		left_fork = 0;
	while(f = food_on_table()){
		printf("dfsdfsdfsdfdfdfdsfsdfsdf\n");
		get_forks((int)num, (int)num);
		printf("philosopher %i get forks\n", (int)num);
		usleep(DELAY * (FOOD - f + 1));
		down_forks(left_fork, rigth_fork);
		printf("philosopher %i eate\n", (int)num);
		counter++;
	}
	printf("dedlock uhodi\n");
	printf("philosopher %i eate %i\n", (int)num, counter);
}

int main(int argc, char *argv[])
{
	int i = 0;	
	
	for(i = 0; i < PHILO; i++){
		pthread_mutex_init(&forks[i], NULL);
	}
	for(i = 0; i < PHILO; i++){
		pthread_create(&phils[i], NULL, philosopher, (void*)i);
	}
	for(i = 0; i < PHILO; i++){
		pthread_join(phils[i], NULL);
	}
	return 0;
}