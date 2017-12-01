#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/stat.h>

#define ITERS 10

typedef struct parametr{
	sem_t *current_sem;
	sem_t *other_sem;
	char *name;
} param;

void *thread_function(void *par){
	param* propertis = (param*) par;
	int i = 0;
	for(i = 0; i < ITERS; i++){
		sem_wait(propertis->current_sem);
		printf("line %d from %s\n", i, propertis->name);
		sem_post(propertis->other_sem);
	}
}

int main(){
	sem_t *child_sem = sem_open("/child", O_CREAT, 775, 0);
	sem_t *parent_sem = sem_open("/parent", O_CREAT, 775, 1);
	param parametr;

	int id = fork();
	if(0 == id){
		parametr.current_sem = child_sem;
		parametr.other_sem = parent_sem;
		parametr.name = "child";
	}
	else{
		parametr.current_sem = parent_sem;
		parametr.other_sem = child_sem;
		parametr.name = "parent";
	}
	thread_function(&parametr);
	sem_close(child_sem);
	sem_close(parent_sem);
	
	if(0 != id){
		sem_unlink("/child");
		sem_unlink("/parent");
	}
	
	return 0;
}