#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <device_functions.h>


#define N 30
#define SIZE 256
#define MIN 0
#define MAX 100
#define DIF (MAX - MIN)

__global__ void kernelFirst(curandState* state, unsigned long seed, int n)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < n)
        curand_init(seed, id, 0, &state[id]);
}

__global__ void kernelGenerate(curandState* gState, float* result, int count)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < count) {
        curandState localState = gState[idx];
        float RANDOM = curand_uniform(&localState);
        gState[idx] = localState;
        result[idx] = RANDOM;
    }
}

void cpuGenetate(float* result, int count)
{
    for (int i = 0; i < count; i++) 
    {
        result[i] = MIN + rand() % MAX + 1;
    }
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    float* dev_result;
    float* result = (float*)malloc(DIF * sizeof(float));
    float* resultCPU = (float*)malloc(DIF * sizeof(float));
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    curandState* devStates;


    if (cudaMalloc(&devStates, DIF * sizeof(curandState)))
        printf("Error: cudaMalloc1\n");
    if (cudaMalloc(&dev_result, DIF * sizeof(float)))
        printf("Error: cudaMalloc2\n");

    kernelFirst << < (DIF + SIZE - 1) / SIZE, SIZE >> > (devStates, time(NULL), DIF);
    if (cudaGetLastError() != cudaSuccess) printf("Error: Kernel");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernelGenerate << < (DIF + SIZE - 1) / SIZE, SIZE >> > (devStates, dev_result, DIF);
    if (cudaGetLastError() != cudaSuccess) printf("Error: Kernel");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("Time GPU = %.2f millseconds\n", gpuTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int start2, time2;
    start2 = clock();

    cpuGenetate(resultCPU, DIF);

    time2 = clock() - start2;
    printf("Time CPU = %d millseconds\n", time2);

    if (cudaMemcpy(result, dev_result, sizeof(float) *DIF, cudaMemcpyDeviceToHost) != cudaSuccess)
        printf("Error: cudaMemcpy3\n");
    printf("Random numbers: \n");
    for (int i = 0; i < N; i++)
        printf("@-> %f\n", result[i]*DIF + MIN);



    return 0;
}