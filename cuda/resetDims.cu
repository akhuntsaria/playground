#include <cuda_runtime.h>
#include <stdio.h>

int calcGridX(int n,dim3 *block){
  return (n+block->x-1)/block->x;
}

int main(){
  int n=512;

  dim3 block(512);
  dim3 grid(calcGridX(n,&block));
  printf("grid: (%d,%d,%d), block: (%d,%d,%d)\n",
      grid.x,grid.y,grid.z,
      block.x,block.y,block.z);

  block.x=256;
  grid.x=calcGridX(n,&block);
  printf("grid: (%d,%d,%d), block: (%d,%d,%d)\n",
      grid.x,grid.y,grid.z,
      block.x,block.y,block.z);

  block.x=128;
  grid.x=calcGridX(n,&block);
  printf("grid: (%d,%d,%d), block: (%d,%d,%d)\n",
      grid.x,grid.y,grid.z,
      block.x,block.y,block.z);

  cudaDeviceReset();
}
