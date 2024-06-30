#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIdx() {
  printf("GPU: threadIdx: (%d,%d,%d), \
      blockIdx: (%d,%d,%d), \
      blocKDim: (%d,%d,%d), \
      gridDim: (%d,%d,%d)\n",
      threadIdx.x,threadIdx.y,threadIdx.z,
      blockIdx.x,blockIdx.y,blockIdx.z,
      blockDim.x,blockDim.y,blockDim.z,
      gridDim.x,gridDim.y,gridDim.z);
}

int main(){
  int n=4096;

  dim3 block(128);
  dim3 grid((n+block.x-1)/block.x);

  printf("CPU: grid: (%d,%d,%d), block: (%d,%d,%d)\n",
      grid.x,grid.y,grid.z,
      block.x,block.y,block.z);

  checkIdx<<<grid,block>>>();

  cudaDeviceReset();
}
