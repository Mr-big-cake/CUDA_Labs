// Лабораторная №1. Вычислить функцию экспоненты

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime> 

__global__ void My_exp(float x, float* arr, int *factorial)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	while (*factorial < idx)
	{
		factorial[*factorial + 1] = (*factorial) * factorial[*factorial];
		(*factorial)++;
	}

	arr[idx] = pow(x, idx) / (float)factorial[idx];
}


void main()
{
	const int order = 15;

	float x;
	printf("###GPU### \nPrint number x, (exp(x)): \n");
	while ((scanf("%f", &x)) != 1) {
		printf("Incoorrect! Try again: ");
		while (getchar() != '\n')
			;
	}

	float* dev_x;
	if (cudaMalloc(&dev_x, sizeof(float)) != cudaSuccess)
		printf("Error: cudaMalloc");
	if (cudaMemcpy(dev_x, &x, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
		printf("Error: cudaMemcpy!");

	float* arr = (float*)malloc(sizeof(float) * order);
	float* dev_arr;
	if (cudaMalloc(&dev_arr, sizeof(float) * order) != cudaSuccess)
		printf("Error: cudaMalloc");

	int* factorial = (int*)malloc(sizeof(int) * (order + 1));

	factorial[0] = 3;
	factorial[1] = 1;
	factorial[2] = 1;
	factorial[3] = 2;



	int* dev_factorial;

	if (cudaMalloc(&dev_factorial, sizeof(int) * (order + 1)) != cudaSuccess) 
		printf("Error: cudaMalloc");

	if (cudaMemcpy(dev_factorial, factorial, sizeof(int) * (order + 1), cudaMemcpyHostToDevice) != cudaSuccess)
		printf("Error: cudaMemcpy");

	int start1, time1;
	start1 = clock();

	My_exp << <1, order >> > (x, dev_arr, dev_factorial);

	if (cudaGetLastError() != cudaSuccess)
		printf("Error: My_exp");

	time1 = clock() - start1;

	if (cudaMemcpy(arr, dev_arr, sizeof(float) * order, cudaMemcpyDeviceToHost) != cudaSuccess)
		printf("Error: cudaMemcpy");

	if (cudaFree(dev_arr) != cudaSuccess)
		printf("Error: cudaFree");

	if (cudaFree(dev_factorial) != cudaSuccess)
		printf("Error: cudaFree");
	float sum = 0;
	for (int i = 0; i < order; i++)
	{
		sum += arr[i];
		//printf("%f ", arr[i]);
	}
	printf("exp(x) = %f", sum);
	char tmp;
	scanf("%c", &tmp);
	/// - CPU
	printf("\n\n###CPU###");
	int start2, time2;
	start2 = clock();
	int n = 2;
	for (int i = 0; i < order; i++)
	{
		if (i == 0) n = 1;
		else  if (i == 1) n = 1;
		else  if (i == 2) n = 2;
		else n *= i;
		arr[i] = pow(x, i) / n;
	}
	time2 = clock() - start2;
	sum = 0;
	for (int i = 0; i < order; i++)
	{
		sum += arr[i];
		//printf("%f ", arr[i]);
	}
	printf("\nexp(x) = %f", sum);
	free(arr);
	free(factorial);


	printf("\n\n\nTimeGPU = %d \nTime CPU = %d",time1, time2);
}