//Сложная 12.	Найти сумму квадратов
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

const int maxThreadsPerBlock = 256;
const int blocksPerGrid = 32;

__global__ void sumOfSquares(float* numbers, float* result, int n)
{
	__shared__ float cache[maxThreadsPerBlock];
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int cacheId = threadIdx.x;
	int totalNumberOfThreads = gridDim.x * blockDim.x;

	float tmp = 0;
	while (idx < n)
	{
		tmp += numbers[idx] * numbers[idx];
		idx += totalNumberOfThreads;
	}

	cache[cacheId] = tmp;
	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheId < i)
			cache[cacheId] += cache[cacheId + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheId == 0)
		result[blockIdx.x] = cache[0];
}

void main()
{
	int n;
	printf("###GPU### \n");
	printf("Print count numbers: \n");
	while ((scanf("%d", &n)) != 1) {
	    printf("Incorrect! Try again: ");
	    while (getchar() != '\n')
	        ;
	}

	float* a, * b, c, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;
	a = (float*)malloc(n * sizeof(float));
	b = (float*)malloc(n * sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

	if (cudaMalloc(&dev_a, n * sizeof(float)) != cudaSuccess)
		printf("Error: cudaMalloc");
	if (cudaMalloc(&dev_b, n * sizeof(float)) != cudaSuccess)
		printf("Error: cudaMalloc");
	if (cudaMalloc(&dev_partial_c, blocksPerGrid * sizeof(float)) != cudaSuccess)
		printf("Error: cudaMalloc");

	//printf("Print %d numbers: \n", n);
	for (int i = 0; i < n; i++)
	{
		float tmp = 0;
		//while ((scanf("%f", &tmp)) != 1) {
		//	printf("Incorrect! Try again: ");
		//	while (getchar() != '\n')
		//		;
		//}
		a[i] = i;
	}

	if (cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
		printf("Error: cudaMemcpy!");
	if (cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
		printf("Error: cudaMemcpy!");


	int start1, time1;
	start1 = clock();
	sumOfSquares << <blocksPerGrid, maxThreadsPerBlock >> > (dev_a, dev_partial_c, n);
	if (cudaGetLastError() != cudaSuccess)
		printf("Error: sumOfSquares");
	time1 = clock() - start1;
	if (cudaMemcpy(partial_c, dev_partial_c, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost) != cudaSuccess)
		printf("Error: cudaMemcpy");

	c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
	}
	printf("Answer: %f", c);
	/// - CPU
	printf("\n\n###CPU###");
	int start2, time2;
	start2 = clock();

	c = 0;
	for (int i = 0; i < n; i++)
	{
		c += a[i] * a[i];
	}
	time2 = clock() - start1;
	printf("\nAnswer: %f", c);

	printf("\n\n\nTimeGPU = %d \nTime CPU = %d", time1, time2);
}