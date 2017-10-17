#include <stdio.h>
#include <stdlib.h>
/*s
функция, которая возвращает куда идти
написать функцию, которая идет вперед
написать функцию, которая идет назад

в функции которая считает размер сделаить проверку на символы и на то что поле прямоугольное.

в функции которая заполняет массив, по которому мы будем ходить найти значение А.

куда идти(катра, текущая позиция, символ, который ищем, размер поля){
есть массив из 4 элементов [10 -10 1 -1],
ПРОБЕГАЕМСЯ ПО элементам этого массива, проверяем:(используя этот элемент не уйдем за границу(для первой координаты первую цифру, для второй вторую, пролучается фор в 4 прохода))
проверяем стоит ли там символ, который мы ищем, если стоит, то возвращаем сразу эту чиселку
конец цикла
возвращаем 0, в мейне не забыть, если куда идти вернула 0, то меняем Б на пробел.
}

возвращает позицию идти вперед(поле, текущая позиция, куда идти){
	идем,
	проверяем не стоим ли мы на Б,
	ставим в клетку куда пришли значение to + 100,
	возвращаем позицию где сейчас стоим.
}

в мейне появляется витвление, если после поиска пустой клетки куда идти вернул 0, то вызываем го бек

идти назад(поле, позиция){
	отнимаем от клетки, в которой стоим 100 
	домножаем на -1
	ставим где сейчас стоим 0
	идем.
}
всю эту штуку повторяется в цикле пока не встанем на Б

усложнение(тараедальность и поиск КРАТЧАЙШЕГО пути)
*/
int get_size(FILE* f, int* lenght, int* height){
	char tmp = fgetc(f);
	int len = 0;
	// tmp = fgetc(f);
	while(!feof(f)){
		len = 0;
		while((tmp != '\n')&&(!feof(f))){
			len++;
			if((*height) == 0){
				(*lenght)++;
			}
			if((tmp != ' ')&&(tmp != '#')&&(tmp != 'A')&&(tmp != 'B')){
				puts("tut");
				return 0;
			}
			tmp = fgetc(f);
		}
		tmp = fgetc(f);
		if(len != (*lenght)){
			puts("zdes");
			return 0;
		}
		(*height)++;
	}
	return 1;
}

char** get_map(FILE* f, int* i_start, int* j_start, int lenght, int height){
	int i, j, s;
	char** map = calloc(height, sizeof(char*));
	for(s = 0; s < lenght; s++){
		map[s] = calloc(lenght, sizeof(char));
	}
	for (i = 0; i < height; i++){
		for(j = 0; j < lenght; j++){
			map[i][j] = fgetc(f);
			if(map[i][j] == 'A'){
				(*i_start) = i;
				(*j_start) = j; 
			}
		}
		fgetc(f);
	}
	return map;
}

int get_next(char** map, int i_now, int j_now, char symbol, int lenght, int height){
	char ways[4] = {10, -1. -10, 1};
	int s;
	for(s = 0; s < 4; s++){
		if((i_now + (ways[s] / 10) < height)&&(i_now + (ways[s] / 10) >= 0)&&(j_now + (ways[s] % 10) < lenght)&&(j_now + (ways[s] % 10) >= 0)){
			if(map[i_now + (ways[s] / 10)][j_now + (ways[s] % 10)] == symbol){
				return ways[s];
			}

		}
	}
	return 0;
}

int go(char** map, int* i_now, int* j_now, int way){
	(*i_now) = ((*i_now) + (way / 10));
	(*j_now) = ((*j_now) + (way % 10));
	if(map[(*i_now)][(*j_now)] != 'B'){
		map[(*i_now)][(*j_now)] = way + 100;
	}
	return 1;
}

int go_back(char** map, int* i_now, int* j_now){
	/*map[*i_now][*j_now] = map[*i_now][*j_now] - 100;
	map[*i_now][*j_now] = map[*i_now][*j_now]*(-1);*/
	if(map[*i_now][*j_now] == 'A')
	{
		return 0;
	}
	int way = 100-map[*i_now][*j_now];
	map[*i_now][*j_now] = '0';
	(*i_now) = ((*i_now) + (way/10));
	(*j_now) = ((*j_now) + (way%10));
	return 1;
}

void print(char** map, int height, int lenght){
	int i,j;
	for(i = 0; i < height; i++){
		for(j = 0; j < lenght; j++){
			if((map[i][j] >= 89)&&(map[i][j] <= 110)){
				printf("*");
			}
			else
			{
				if(map[i][j] == '0'){
					printf(" ");
				}
				else{
					printf("%c", map[i][j]);
				}
			}
		}
		printf("\n");
	}
}

int main(int argc, char** argv){
	FILE* f = fopen(argv[1], "r");
	int lenght = 0, height = 0, s = 0, i_start, j_start, i_now, j_now, to;
	if(get_size(f, &lenght, &height) == 0){
		puts("incorrect file");
		return 0;
	}
	char** map;
	fclose(f);
	f = fopen(argv[1], "r");
	map = get_map(f, &i_start, &j_start, lenght, height);
	i_now = i_start;
	j_now = j_start;
	while(map[i_now][j_now] != 'B'){
		print(map,height,lenght);
		printf("\n");
		to = get_next(map, i_now, j_now, 'B', lenght, height);
		if(to == 0){
			to = get_next(map, i_now, j_now, ' ', lenght, height);
		}
		if(to != 0){
			go(map, &i_now, &j_now, to);
		}
		else{
			if(go_back(map, &i_now, &j_now)==0)
			{
				printf("no exit\n");
				return 0;
				
			}
		}
	}
	print(map, height, lenght);
} 