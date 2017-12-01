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
	pthread_rwlock_t rwlock;

	LinkedList(){
		this->head = NULL;
		pthread_rwlock_init(&rwlock, NULL);
	}
	~LinkedList(){
		Elem *p = head;

		while(p != NULL){
			Elem *q = p;
			p = p->next;
			delete q;
		}

		pthread_rwlock_destroy(&rwlock);

	}
	
	void wrlock(){
		pthread_rwlock_wrlock(&rwlock);
	}

	void rdlock(){
		pthread_rwlock_rdlock(&rwlock);
	}

	void unlock(){
		pthread_rwlock_unlock(&rwlock);
	}

	void push_front(char *str){
		wrlock();

		head = new Elem(str, head);

		unlock();
	}

	void print(){
		rdlock();

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
		wrlock();
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