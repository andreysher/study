#define N 5
#define W 13
#include <stdio.h>

int ThingsTable[N][2];

ThingsTable[1][0] = 

int A[N][W];

void FindAnser(int ThingNumber, int capacity){
	if (A[ThingNumber][capacity] == 0){
		return;
	}
	if (A[ThingNumber-1][capacity] == A[ThingNumber][capacity]){
		FindAnser(ThingNumber-1, capacity);
	}
	else{
		printf ("%i", ThingNumber);
		FindAnser(ThingNumber-1, capacity - ThingsTable[ThingNumber][0]);
	}
}

int Max(int a, int b){
	if(a >= b)
		return a;
	
	else
		return b;
	}



int main(){

	for (int i = 0; i <= W; i++){
		A[0][i] = 0;
	}
	for (int i = 0; i <= N; i++){
		A[i][0] = 0;
	}
	
	for (int k = 1; k <= N; k++){ 
		for (int s = 1; s <= W; s++){   //Перебираем для каждого k все вместимости 
			if (s >= ThingsTable[k][0]){    //Если текущий предмет вмещается в рюкзак
				A[k][s] = Max(A[k-1][s], A[k-1][s-ThingsTable[k][0]]+ThingsTable[k][1]); //выбираем класть его или нет
			}
			else{ 
				A[k][s] = A[k-1][s]; 
			}
		}
	}
	
	FindAnser(5, 13);
	
	return 0;
}