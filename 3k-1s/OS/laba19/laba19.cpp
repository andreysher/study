#include "LinkedList.cpp"

#define BUFSIZE 81

void *childFunction(void *param){
	LinkedList *list = (LinkedList*)param;
	while(1){
		sleep(5);
		(*list).sort();
	}
}

int main(int argc, char const *argv[])
{
	std::cout << "\tplease type some text, \n \
	80 characters is maximum, it will \n \
	be sorted each 5 seconds\n \
	for check enter empty string,\n \
	for stop program type \'stop\' \n";

	LinkedList list;

	pthread_t child;
	for(int i = 0; i < 100; i++){
		pthread_create(&child, NULL, childFunction, &list);
	}
	char buffer[BUFSIZE] = {0};

	printf("> ");

	while((NULL != fgets(buffer, BUFSIZE, stdin)) && (0 != strcmp(buffer, "stop\n"))){

		if(1 == strlen(buffer)){
			list.print();
		}
		else{
			list.push_front(buffer);
		}
		printf("> ");
	}
	return 0;
}