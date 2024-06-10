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

  float *a,*b,*c;
  a=(float *)malloc(bytes);
  b=(float *)malloc(bytes);
  c=(float *)malloc(bytes);

  init(a,n);
  init(b,n);

  sum(a,b,c,n);
  printf("%f+%f=%f",a[0],b[0],c[0]);

  free(a);
  free(b);
  free(c);
	return(0);
}
