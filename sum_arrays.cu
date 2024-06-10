#include <stdlib.h>
#include <string.h>
#include <time.h>

void sum(float *a, float *b, float *c, const int n) {
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
  int n=2048;
  size_t bytes=n*sizeof(float);

  float *ha,*hb,*hc;
  a=(float *)malloc(bytes);
  b=(float *)malloc(bytes);
  c=(float *)malloc(bytes);

  float *da,*db,*dc;
  cudaMalloc(&da, bytes);
  cudaMalloc(&db, bytes);
  cudaMalloc(&dc, bytes);

  init(a,n);
  init(b,n);

  cudaMemcpy(da,ha,n,cudaMemcpyHostToDevice);
  cudaMemcpy(db,hb,n,cudaMemcpyHostToDevice);

  //...

  cudaMemcpy(hc,dc,n,cudaMemcpyHostToDevice);

  printf("%f+%f=%f",ha[0],hb[0],hc[0]);

  free(a);
  free(b);
  free(c);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
	return(0);
}
