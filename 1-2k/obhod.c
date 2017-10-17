#include <stdio.h>
#include <malloc.h>
//#include "graph_lib.h"

//для каждой задачи задается своя в зависимости от требований, имеет поля number, name, weigth, flag...
typedef struct graph_element{
	int number;
	struct graph_element* next;
	int weigth;
}graph_el;

void* GraphCreate(FILE* file, int* size_graph){// указатель куда положить размер, разыменовывается и кладется размер
	if(file == NULL){
		return NULL;
	}
	int size = 0, x = 0, y = 0, i = 0;
	char type;//"m" или "l" 
	char weigth;//"1" или "0"
	char orient;//"1" или "0"
	if (fscanf(file, "%c %c %c", &type, &weigth, &orient) == EOF){
		printf("err");
	}
	fscanf(file, "%i", &size);//следует ли перед fscanf переходить на строчку, или он будет делать fscanf пока не найдет нужную чиселку(всмысле формат данных)
	*size_graph = size;//тут тоже не уверен
	void* result;
	if(type == 'm'){
		int** Matrix = calloc(size, sizeof(int*));
		for(i = 0; i < size; i++){
			*(Matrix + i) = calloc(size, sizeof(int));
		}
		if((orient == '1')&&(weigth == '0')){
			while (fscanf(file, "%i %i", &x, &y) != EOF) { //??
				Matrix[x-1][y-1] = 1;
			}//знаю, что здесь наговнокодил, надо было рассматривать случаи когда взвешаный, ориентированый, не взвешаный и не ориентированый по отдельности
		}
		if((orient == '0')&&(weigth == '0')){
			while (fscanf(file, "%i %i", &x, &y) != EOF) { //??
				Matrix[x-1][y-1] = 1;
				Matrix[y-1][x-1] = 1;
			}
		}
		if((orient == '1')&&(weigth == '1')){
			int wt = 1;
			while (fscanf(file, "%i %i %i", &x, &y, &wt) != EOF) { //??
				Matrix[x-1][y-1] = wt;
			}
		}
		if((orient == '0')&&(weigth == '1')){
			int wt = 1;
			while (fscanf(file, "%i %i %i", &x, &y, &wt) != EOF) { //??
				Matrix[x-1][y-1] = wt;
				Matrix[y-1][x-1] = wt;
			}
		}
		result = Matrix; // нужно ли явное преобразование типов указателей?
	}
		if(type == 'l'){
			graph_el* List = calloc(size, sizeof(graph_el));
			if((weigth == '0')&&(orient == '1')){
				fscanf(file, "%i %i", &x, &y);
				if(List[x].number == 0){
					List[x].number = y; 
				}
				if(List[x].number != 0){
					graph_el* go;
					go = &List[x];
					while(go->next != NULL){
						go = go->next;
					}
					graph_el* new = calloc(1, sizeof(graph_el));
					new->number = y;
					go->next = new;
				}
			}
			if((weigth == '1')&&(orient == '1')){
				int wt = 1;
				fscanf(file, "%i %i %i", &x, &y, &wt);
				if(List[x].number == 0){
					List[x].number = y;
					List[x].weigth = wt;
				}
				if(List[x].number != 0){
					graph_el* go = calloc(1, sizeof(graph_el));
					go = &List[x];
					while(go->next != NULL){
						go = go->next;
					}
					graph_el* new = calloc(1, sizeof(graph_el));
					new->number = y;
					new->weigth = wt;
					go->next = new;
			}
			if((weigth == '0')&&(orient == '0')){
				fscanf(file, "%i %i", &x, &y);
				if(List[x].number == 0){
					List[x].number = y; 
				}
				if(List[x].number != 0){
					graph_el* go = calloc(1, sizeof(graph_el));
					go = &List[x];
					while(go->next != NULL){
						go = go->next;
					}
					graph_el* new = calloc(1, sizeof(graph_el));
					new->number = y;
					go->next = new;
				}
				if(List[y].number == 0){
					List[y].number = x; 
				}
				if(List[y].number != 0){
					graph_el* go = calloc(1, sizeof(graph_el));
					go = &List[y];
					while(go->next != NULL){
						go = go->next;
					}
					graph_el* new = calloc(1, sizeof(graph_el));
					new->number = x;
					go->next = new;
				}
			}
			if((weigth == '1')&&(orient == '0')){
				int wt = 1;
				fscanf(file, "%i %i %i", &x, &y, &wt);
				if(List[x].number == 0){
					List[x].number = y;
					List[x].weigth = wt;
				}
				if(List[x].number != 0){
					graph_el* go = calloc(1, sizeof(graph_el));
					go = &List[x];
					while(go->next != NULL){
						go = go->next;
					}
					graph_el* new = calloc(1, sizeof(graph_el));
					new->number = y;
					new->weigth = wt;
					go->next = new;
				}
				if(List[y].number == 0){
					List[y].number = x;
					List[y].weigth = wt; 
				}
				if(List[y].number != 0){
					graph_el* go = calloc(1, sizeof(graph_el));
					go = &List[y];
					while(go->next != NULL){
						go = go->next;
					}
					graph_el* new = calloc(1, sizeof(graph_el));
					new->number = x;
					new->weigth = wt;
					go->next = new;
				}
			}
		void* result = List;
		}
			}
	return result;
}

void glubina(int size, int** Graph, int* Visited, int Node) {
	Visited[Node] = 1;
	int i = 0;
	printf("%i ", Node + 1);
	for (i = 0; i < size; i++) {
		if ((Graph[Node][i]==1) && (Visited[i]!=1)) {
			glubina(size, Graph, Visited, i);
		}
	}
	return;
}
void shirina(int size, int** Graph, int *Visited, int Node) {
	int* List = (int*)calloc(size, sizeof(int));
	int Count, Head;
	int i;
	Count = Head = 0;
	List[Count++] = Node;
	Visited[Node] = 1;
	while (Head < Count) {
		Node = List[Head++];
		printf("%i", Node + 1);
		for (i = 0; i < size; i++)
			if (Graph[Node][i] && !Visited[i]) {
				List[Count++] = i;
				Visited[i] = 1;
			}
	}
}

int main (int argc, char* argv[]) {
	FILE* f = fopen(argv[1], "r");
	int* Visited;
	int size = 0, z = 0;
	Visited = (int*)calloc(size, sizeof(int));
	void* Graph = GraphCreate(f, &size);// по идее size изменилась
	glubina(size, (int**)Graph, Visited, 0);
	for (z = 0; z < size; z++){
		Visited[z] = 0;
	}
	printf("\n");
	shirina(size, (int**)Graph, Visited, 0);
	printf("\n");
	return 0;
}