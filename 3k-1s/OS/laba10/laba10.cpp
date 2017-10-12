#include <cstdio>
#include <pthread.h>
#include <unistd.h>

#define QUEUESIZE 100

pthread_mutex_t *queue;

bool started = false;

void * threadFunction(void * parameter)
{
    char * threadName = (char *)parameter;

    int index = 2;

    pthread_mutex_t *currentMutex = &queue[index - 1];
    pthread_mutex_t *previousMutex = NULL;

    if (0 != pthread_mutex_lock(currentMutex))//stop point
    {
        perror("Error while initial pthread_mutex_lock().\n");

        return NULL;
    }

    if (started)
    {
        if (0 != pthread_mutex_unlock(&queue[0]))
        {
            perror("Error while started pthread_mutex_unlock().\n");

            return NULL;
        }
    }

    for (int i = 0; i < 10; i++)
    {
        for ( ; index < QUEUESIZE; index++)
        {
            previousMutex = currentMutex;
            currentMutex = &queue[index];

            if (0 != pthread_mutex_lock(currentMutex))
            {
                perror("Error while pthread_mutex_lock(current).\n");
            }

            if (0 != pthread_mutex_unlock(previousMutex))
            {
                perror("Error while pthread_mutex_unlock(previousMutex).\n");
            }

            if (QUEUESIZE - 1 == index)
            {
                printf("Line %d from thread %s\n", i, threadName);

                started = true;
            }
        }

        index = 0;
    }

    if (0 != pthread_mutex_unlock(currentMutex))
    {
        perror("Error while final pthread_mutex_unlock().\n");
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    queue = new pthread_mutex_t[QUEUESIZE];

    for (int i = 0; i < QUEUESIZE; i++)
    {
        if (0 != pthread_mutex_init(&queue[i], NULL))
        {
            perror("Error while pthread_mutex_init().\n");
        }
    }

    pthread_t thread;

    char childName[] = "child";

    if (0 != pthread_create(&thread, NULL, threadFunction, childName))
    {
        perror("Error while perror_create.\n");
    }
    else
    {
        pthread_mutex_lock(&queue[0]);

        while (!started)
        {
            if (0 != sleep(1))
            {
                perror("Sleep was interrupted");
            }
        }

        char parentName[] = "parent";

        threadFunction(&parentName);

        if (0 != pthread_join(thread, NULL))
        {
            perror("Error while pthread_join.\n");
        }
    }

    for (int i = 0; i < QUEUESIZE; i++)
    {
        if (0 != pthread_mutex_destroy(&queue[i]))
        {
            perror("Error while pthread_mutex_destroy().\n");
        }
    }

    delete[] queue;

    return 0;
}