#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void *f(){

}

int main(int argc, char const *argv[])
{
	printf("%d\n",sizeof(pthread_t));
	return 0;
}