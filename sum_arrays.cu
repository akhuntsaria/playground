#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void sumDev(float *a,float *b,float *c){
  int i=threadIdx.x;
  printf("Thread %d\n",i);
  c[i]=a[i]+b[i];
}

void sumHost(float *a, float *b, float *c, const int n) {
	for(int i=0;i<n;i++){
    c[i]=a[i]+b[i];
  }
}

void init(float *a,int n){
  time_t t;
  srand((unsigned int) time(&t));

  for(int i=0;i<n;i++){
    a[i]=(rand()&0xFF)/10.0f;
  }
}

int main(int argc, char **argv) {
  int n=32;
  size_t bytes=n*sizeof(float);

  float *ha,*hb,*hc;
  ha=(float *)malloc(bytes);
  hb=(float *)malloc(bytes);
  hc=(float *)malloc(bytes);

  float *da,*db,*dc;
  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);
  cudaMalloc(&dc, bytes);

  init(ha,n);
  init(hb,n);

  cudaMemcpy(da,ha,n,cudaMemcpyHostToDevice);
  cudaMemcpy(db,hb,n,cudaMemcpyHostToDevice);

  sumDev<<<1,n>>>(da,db,dc);

  cudaMemcpy(hc,dc,n,cudaMemcpyHostToDevice);

  printf("%f+%f=%f\n",ha[0],hb[0],hc[0]);

  free(ha);
  free(hb);
  free(hc);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
	return(0);
}
