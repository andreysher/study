#define INF 65000
#include "graph_lib.h"

int shortest(int from, Graph grp, int* obrabotan, int* visit){
	int i, min_index, min = INF;
	for(i = 0; i < Give_Size(grp); i++){
		if((Give_Weigth_connection(grp, from, i) > 0)&&(Give_Weigth_connection(grp, from, i) < min)&&(obrabotan[i] == 0)&&(visit[i] == 0)){
				min = Give_Weigth_connection(grp, from, i);
				//printf("%i\n", min);
				min_index = i;
		}
	}
	return min_index;
}

int deykstra(int start, int finish, Graph my_graph){
	int *path = calloc(Give_Size(my_graph), sizeof(int));
	int *befor = calloc(Give_Size(my_graph), sizeof(int));
	int curent, i, o, z, j, count = 0;
	int* visit = calloc(Give_Size(my_graph), sizeof(int));
	int* obrabotan = calloc(Give_Size(my_graph), sizeof(int));
	for(i = 0; i < Give_Size(my_graph); i++){
		path[i] = INF;
	}
	path[start] = 0;
	i = 0;
	while(i != Give_Size(my_graph)){//идем по всем вершинам
		curent = start;//начинаем со страта
		for(j = 0; j < Give_Size(my_graph); j++){//для каждой вершины обрабатываем ее соседей
			count = 0;
			if(Give_Weigth_connection(my_graph, curent, j) == 0){
				obrabotan[j] = 1;
				count++;
			}
			//printf("%i %i", path[curent], shortest(curent, my_graph, obrabotan, visit));
			while(count != Give_Size(my_graph)){
				if((path[curent] + Give_Weigth_connection(my_graph, curent, shortest(curent, my_graph, obrabotan, visit))) < path[shortest(curent, my_graph, obrabotan, visit)]){
					path[shortest(curent, my_graph, obrabotan, visit)] = path[curent] + Give_Weigth_connection(my_graph, curent, shortest(curent, my_graph, obrabotan, visit));
				}
				obrabotan[shortest(curent, my_graph, obrabotan, visit)] = 1;
				count++;
			}
		}
		visit[curent] = 1;//отмечаем данную вершину как посещенную
		for(z = 0; z < Give_Size(my_graph); z++){
			obrabotan[z] = 0;
		}
		curent = shortest(curent, my_graph, obrabotan, visit);//перешли на следующую
		//printf("%i",curent);
		i++;
	}
	/*for( o = 0; o < Give_Size(my_graph); o++){
		printf("%i",path[o]);
	}*/
	return path[finish];
}

int main(int argc, char *argv[])
{
	FILE* file = fopen(argv[1], "r");
	int start, finish;
	Graph my_graph = Download_from_file(file);
	while(1){
		scanf("%i %i", &start, &finish);
		if((start == -1)||(finish == -1)){
			return;
		}
		printf("%i\n", deykstra(start, finish, my_graph));
	}
}