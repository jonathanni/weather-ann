#define __LIMIT__ 65536

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>

int * a = 0;
int * b = 0;
int * c = 0;

int * a1 = 0;
int * b1 = 0;
int * c1 = 0;

void callKernel();
void cudaRead();
void cudaWrite();
void cudaSync();

__global__ void add(int lim, int * a, int * b, int * c)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < lim)
    c[n] = a[n] + b[n];
}

void randfill(int * arr, int lim)
{
  int i = 0;
  for(i = 0; i < lim; i++)
    arr[i] = rand();
}

int main()
{
  srand(time(NULL));

  a1 = (int *) calloc(sizeof(int), __LIMIT__);
  b1 = (int *) calloc(sizeof(int), __LIMIT__);
  c1 = (int *) calloc(sizeof(int), __LIMIT__);

  cudaMalloc((void **) &a, __LIMIT__ * sizeof(int));
  cudaMalloc((void **) &b, __LIMIT__ * sizeof(int));
  cudaMalloc((void **) &c, __LIMIT__ * sizeof(int));
  randfill(a1, __LIMIT__);
  randfill(b1, __LIMIT__);
 
  callKernel();

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  free(a1);
  free(b1);
}

void callKernel()
{
  cudaRead();
  add<<<(__LIMIT__ + 255) / 256, 256>>>(__LIMIT__, a, b, c);
  cudaSync();
  cudaWrite();
}

void cudaSync()
{
  cudaError(cudaDeviceSynchronize());
}

void cudaRead()
{
  cudaError(cudaMemcpy(a, a1, __LIMIT__ * sizeof(int), cudaMemcpyHostToDevice));
  cudaError(cudaMemcpy(b, b1, __LIMIT__ * sizeof(int), cudaMemcpyHostToDevice));
}

void cudaWrite()
{
  cudaError(cudaMemcpy(c1, c, __LIMIT__ * sizeof(int), cudaMemcpyDeviceToHost));
}

void cudaError(cudaError_t err)
{
  if(err != cudaSuccess)
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
}
