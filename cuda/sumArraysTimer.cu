#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../windows/gettimeofday.h"

#define CHECK(call) {\
  const cudaError_t error = call; \
  if (error != cudaSuccess) { \
    printf("Error: %s:%d, code: %d. reason: %s\n", __FILE__, __LINE__, error, \
        cudaGetErrorString(error)); \
    exit(1); \
  } \
}

void compare(float *a,float *b,int n){
  double e=1.0E-8;
  for(int i=0;i<n;i++){
    if(abs(a[i]-b[i])>e){
      printf("Arrays don't match\n");
      printf("i: %d; %4.2f != %4.2f\n",i,a[i],b[i]);
      return;
    }
  }
  printf("Arrays match\n");
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void init(float *a,int n){
  time_t t;
  srand((unsigned int) time(&t));

  for(int i=0;i<n;i++){
    a[i]=(rand()&0xFF)/10.0f;
  }
}

__global__ void sumDev(float *a, float *b, float *c, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];

}

void sumHost(float *a, float *b, float *c, const int n) {
  for(int i=0;i<n;i++){
    c[i]=a[i]+b[i];
  }
}

int main(int argc, char **argv) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Using device %d: %s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

  int n=1<<28;
  printf("Vector size: %d\n", n);

  size_t bytes=n*sizeof(float);

  float *ha,*hb,*hc,*hd;
  ha=(float *)malloc(bytes);
  hb=(float *)malloc(bytes);
  hc=(float *)malloc(bytes);
  hd=(float *)malloc(bytes);

  float *da,*db,*dc;
  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);
  cudaMalloc(&dc, bytes);

  init(ha,n);
  init(hb,n);

  double iStart = cpuSecond(); 

  sumHost(ha,hb,hc,n);

  double hostElaps = cpuSecond() - iStart;
  printf("sumHost, time elapsed: %f sec\n", hostElaps);

  cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);

  int iLen = 1024;
  dim3 block(iLen);
  dim3 grid((n + block.x - 1) / block.x);

  iStart = cpuSecond();
  
  sumDev <<<grid, block>>>(da, db, dc, n);
  CHECK(cudaDeviceSynchronize());
  
  double devElaps = cpuSecond() - iStart;
  printf("sumDev <<<%d, %d>>>, time elapsed: %f sec\n", grid.x, block.x, devElaps);

  printf("%.0fx faster\n", round(hostElaps / devElaps));

  cudaMemcpy(hd, dc, bytes, cudaMemcpyDeviceToHost);

  compare(hc,hd,n);

  free(ha);
  free(hb);
  free(hc);
  free(hd);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return(0);
}
