#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <ctime> 


#define countThreads 1024
#define countBlocks 9
#define N 900;

__constant__ long long dev_constN;
__global__ void _Perfect_GPU(long long * result, long long maxValue)
{
	long long idx = blockDim.x * blockIdx.x + threadIdx.x + 2;
	if (idx > maxValue) return;
	long long sum = 0;
	*result = 0;
	for (long long j = 1; j < idx; j++)
		if (idx % j == 0)
			sum += j;
	if (sum == idx) {
		printf("GPU-->%d\n", idx);
		result[1 + *result] = idx;
	}
}

void _Perfect_CPU(long long n)
{
	long long i, j, s;
	for (i = 2; i < n; i++)
	{
		s = 0;
		for (j = 1; j < i; j++)
			if (i % j == 0)
				s += j;
		if (s == i)
			printf("CPU-->%lld\n", i);
	}
}

void main()
{
	long long* result = (long long*) malloc(sizeof(long long) * 5 );
	long long* dev_result;

	long long constN = N;


	if (cudaMemcpyToSymbol(dev_constN, &constN, sizeof(constN), 0, cudaMemcpyHostToDevice) != cudaSuccess)
		printf("Error: cudaMemcpyToSymbol\n");
	
	if (cudaMalloc(&dev_result, sizeof(long long) * 5) != cudaSuccess)
		printf("Error: cudaMalloc\n");


	if (cudaGetLastError() != cudaSuccess)
		printf("Error: Kernel\n");

	cudaEvent_t start, stop; 
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0); 

	_Perfect_GPU <<<countBlocks, countThreads >> > (dev_result, constN);

	if (cudaGetLastError() != cudaSuccess)
		printf("Error: Kernal");

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&gpuTime, start, stop); 
	printf("Time GPU = %.2f millseconds\n", gpuTime);
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);

	if (cudaMemcpy(result, dev_result, sizeof(long long) * 5, cudaMemcpyDeviceToHost) != cudaSuccess)
		printf("Error: cudaMemcpy2\n");

	int start2, time2;
	start2 = clock();

	_Perfect_CPU(9000);

	time2 = clock() - start2;
	printf("Time CPU = %d millseconds\n", time2);
}
