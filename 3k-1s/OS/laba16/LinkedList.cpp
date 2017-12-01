#include <cstring>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string.h>

class LinkedList
{
public:
	class Elem{
		public:
			char *data;
			Elem *next;
			Elem(char *str, Elem *prevHead){
				char* pointer;
				while((pointer = strchr(str,27)) != NULL){
					*pointer = '?';
				}
				this->data = strdup(str);
				this->next = prevHead;
			}
			~Elem(){
				delete[] data;
			}
	};

	Elem *head;
	pthread_mutex_t mutex;

	LinkedList(){
		this->head = NULL;
		pthread_mutex_init(&mutex, NULL);
	}
	~LinkedList(){
		Elem *p = head;

		while(p != NULL){
			Elem *q = p;
			p = p->next;
			delete q;
		}

		pthread_mutex_destroy(&mutex);

	}
	
	void lock(){
		pthread_mutex_lock(&mutex);
	}

	void unlock(){
		pthread_mutex_unlock(&mutex);
	}

	void push_front(char *str){
		lock();

		head = new Elem(str, head);

		unlock();
	}

	void print(){
		lock();

		int i = 0;
		for (Elem *p = head; p != NULL; p = p->next, i++)
		{
			printf("%s", p->data);
		}

		unlock();
	}

	void sort(){
		if(head == NULL){
			return;
		}
		lock();
		int menyali = 1;
		while(menyali){
			menyali = 0;
			Elem *prev = NULL;
			Elem *i = head;
			while(i->next != NULL){
				if(strcmp(i->data, i->next->data) > 0){
					if(prev != NULL){
						prev->next = i->next;
					}
					else{
						head = i->next;
					}

					prev = i->next;
					i->next = i->next->next;
					prev->next = i;

					menyali = 1;
				}
				else{
					prev = i;
					i = i->next;
				}
			}
		}
		unlock();
	}

};