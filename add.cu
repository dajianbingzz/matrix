#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
using namespace std;
__global__ void vectorADD(const float*A,const float*B,float*C,int numElements)
{
   int i=blockDim.x*blockIdx.x+threadIdx.x;
   if(i<numElements)
   {
    printf("a:%f\n",A[i]);
    printf("b:%f\n",B[i]);
    C[i]=A[i]+B[i];
    printf("c:%f\n",C[i]);
   }
}
int main()
{
     //cpu定义
    int size=1<<20;
    int bytes=size*sizeof(float);

    float *h_a, *h_b,*h_c;
    h_a=(float*)malloc(bytes);
    h_b=(float*)malloc(bytes);
    h_c=(float*)malloc(bytes);
    for(int i=0;i<size;i++)
    {    
    h_a[i]=i+1;
    h_b[i]=2*i+1;
    }
//GPU定义
    float *d_a,*d_b,*d_c;
    //申请GPU上的三个空间
    cudaMalloc((void**)&d_a,bytes);
    cudaMalloc((void**)&d_b,bytes);
    cudaMalloc((void**)&d_c,bytes);
    //cup传GPU
    printf("Copy input data from host to device memory\n");
    cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice);
//执行GPU kernel函数
    int blocksize=256;
    int gridsize=(size+blocksize-1)/blocksize;
    printf("%d \n",gridsize);
    vectorADD<<<gridsize,blocksize>>>(d_a,d_b,d_c,size);
   
   
   
    cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost);

    //验证
    for(int i=0;i<size;++i)
    {
       // printf("h_a=%f,h_b=%f,h_c=%f",h_a[i], h_b[i],h_c[i]);
        if(fabs(h_a[i]+h_b[i]-h_c[i])>1e-5){
            printf("Result verification failed at element%d\n",i);
             exit(EXIT_FAILURE);
        }
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Free(h_a);
    // Free(h_b);
    // Free(h_c);
    printf("test pass\n");
    return 0;
}
