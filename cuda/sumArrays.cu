#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(call) {\
  const cudaError_t error = call; \
  if (error != cudaSuccess) { \
    printf("Error: %s:%d, code: %d. reason: %s\n", __FILE__, __LINE__, error, \
        cudaGetErrorString(error)); \
    exit(2); \
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

void init(float *a,int n){
  time_t t;
  srand((unsigned int) time(&t));

  for(int i=0;i<n;i++){
    a[i]=(rand()&0xFF)/10.0f;
  }
}

__global__ void sumDev(float *a,float *b,float *c){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = a[i] + b[i];
  //printf("Thread index %3d; a[i]:\t%.2f\n", i, a[i]);

  // Cause an error
  //int* ptr = (int*) 0x12345678; // invalid memory address
  //ptr[0] = 0; // will cause a segmentation fault
}

void sumHost(float *a, float *b, float *c, const int n) {
  for(int i=0;i<n;i++){
    c[i]=a[i]+b[i];
  }
}

int main(int argc, char **argv) {
  int n=1024;
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
  sumHost(ha,hb,hc,n);

  cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);

  dim3 block(n);
  dim3 grid(n/block.x);

  sumDev<<< grid,block >>>(da, db, dc);
  CHECK(cudaDeviceSynchronize());

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
