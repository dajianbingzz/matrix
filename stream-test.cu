#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>  
#include <cuda_runtime.h>
using namespace std;
#define N (1024*1024)//每个流的大小
#define FULL (N*20)//全部数据大小
__global__ void kernel_add(int *a,int *b,int*c)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    //int idy=blockdim.y*blockId.y+threadId.y;
    c[idx]=a[idx]+b[idx];
}
int main()
{
    //查询设备属性
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop,whichDevice);
    if(!prop.deviceOverlap)
    {
        printf("Device will not support\n");
        return 0;
    }
    else 
    {printf("device pass\n");}

    //初始化计时器事件
    cudaEvent_t start,stop;
    float elapsedTime;
    //申明流和buffer指针
    cudaStream_t stream0,stream1,stream2,stream3;
    int *a_d,*b_d,*c_d;
    int *a_h,*b_h,*c_h;
        
    int *a1_d,*b1_d,*c1_d;
    int *a2_d,*b2_d,*c2_d;
    int *a3_d,*b3_d,*c3_d;
    //申请GPU空间
    cudaMalloc((void**)&a_d,N*sizeof(int));
    cudaMalloc((void**)&b_d,N*sizeof(int));
    cudaMalloc((void**)&c_d,N*sizeof(int));
    
    cudaMalloc((void**)&a1_d,N*sizeof(int));
    cudaMalloc((void**)&b1_d,N*sizeof(int));
    cudaMalloc((void**)&c1_d,N*sizeof(int));

    cudaMalloc((void**)&a2_d,N*sizeof(int));
    cudaMalloc((void**)&b2_d,N*sizeof(int));
    cudaMalloc((void**)&c2_d,N*sizeof(int));

    cudaMalloc((void**)&a3_d,N*sizeof(int));
    cudaMalloc((void**)&b3_d,N*sizeof(int));
    cudaMalloc((void**)&c3_d,N*sizeof(int));
    //在cpu端申请内存空间，用锁页内存
    cudaHostAlloc((void**)&a_h,FULL*sizeof(int),cudaHostAllocDefault);
    cudaHostAlloc((void**)&b_h,FULL*sizeof(int),cudaHostAllocDefault);
    cudaHostAlloc((void**)&c_h,FULL*sizeof(int),cudaHostAllocDefault);
    
    //创建时间寄存器
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   
    //初始化流
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);   
    //初始化A，B向量
    for(int i=0;i<FULL;i++)
    {
        a_h[i]=i;
        b_h[i]=2*i;
    }
  
    //开始计算1个流的方法
    cudaEventRecord(start,0);
    for(int i=0;i<FULL;i+=N)
    {
        //CPU->GPU
        cudaMemcpyAsync(a_d,a_h+i,N*sizeof(int),cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(b_d,b_h+i,N*sizeof(int),cudaMemcpyHostToDevice,stream0);
       kernel_add<<<N/256,256,0,stream0>>>(a_d,b_d,c_d);
       //GPU->CPU
       cudaMemcpyAsync(c_h+i,c_d,N*sizeof(int),cudaMemcpyDeviceToHost,stream0);
    }
    
    cudaStreamSynchronize(stream0);
    cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("time 1 stream=%fms\n",elapsedTime);
//使用两个流的方法
    cudaEventRecord(start,0);
    for(int i=0;i<FULL;i+=2*N)
    {
        cudaMemcpyAsync(a_d,a_h+i,N*sizeof(int),cudaMemcpyHostToDevice,stream0);           
        cudaMemcpyAsync(a1_d,a_h+i+N,N*sizeof(int),cudaMemcpyHostToDevice,stream1);
        cudaMemcpyAsync(b_d,b_h+i,N*sizeof(int),cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(b1_d,b_h+i+N,N*sizeof(int),cudaMemcpyHostToDevice,stream1);   
        kernel_add<<<N/256,256,0,stream0>>>(a_d,b_d,c_d);
        kernel_add<<<N/256,256,0,stream1>>>(a1_d,b1_d,c1_d);   
        cudaMemcpyAsync(c_h+i,c_d,N*sizeof(int),cudaMemcpyDeviceToHost,stream0);
        cudaMemcpyAsync(c_h+i+N,c1_d,N*sizeof(int),cudaMemcpyDeviceToHost,stream1);
    } 
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaEventRecord(stop,0);
   // cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("time 2 stream=%fms\n",elapsedTime);
    //验证
    for(int i=0;i<FULL;i++)
    {
        if(c_h[i]-a_h[i]-b_h[i]!=0)
        {
            printf("wrong at:%d\n",i);
             exit(EXIT_FAILURE);
        }
        
    }
    printf("pass\n");
    //使用四个流的方法
    // cudaEventRecord(start,0);
    // for(int i=0;i<FULL;i+=4*N)
    // {
    //     cudaMemcpyAsync(a_d,a_h+i,N*sizeof(int),cudaMemcpyHostToDevice,stream0);           
    //     cudaMemcpyAsync(a1_d,a_h+i+N,N*sizeof(int),cudaMemcpyHostToDevice,stream1);
    //     cudaMemcpyAsync(a2_d,a_h+i+2*N,N*sizeof(int),cudaMemcpyHostToDevice,stream2);
    //     cudaMemcpyAsync(a3_d,a_h+i+3*N,N*sizeof(int),cudaMemcpyHostToDevice,stream3);

    //     cudaMemcpyAsync(b_d,b_h+i,N*sizeof(int),cudaMemcpyHostToDevice,stream0);
    //     cudaMemcpyAsync(b1_d,b_h+i+N,N*sizeof(int),cudaMemcpyHostToDevice,stream1); 
    //     cudaMemcpyAsync(b2_d,b_h+i+2*N,N*sizeof(int),cudaMemcpyHostToDevice,stream2);
    //     cudaMemcpyAsync(b3_d,b_h+i+3*N,N*sizeof(int),cudaMemcpyHostToDevice,stream3);  
       
    //     kernel_add<<<N/256,256,0,stream0>>>(a_d,b_d,c_d);
    //     kernel_add<<<N/256,256,0,stream1>>>(a1_d,b1_d,c1_d);
    //     kernel_add<<<N/256,256,0,stream2>>>(a2_d,b2_d,c2_d); 
    //     kernel_add<<<N/256,256,0,stream3>>>(a3_d,b3_d,c3_d); 

    //     cudaMemcpyAsync(c_h+i,c_d,N*sizeof(int),cudaMemcpyDeviceToHost,stream0);
    //     cudaMemcpyAsync(c_h+i+N,c1_d,N*sizeof(int),cudaMemcpyDeviceToHost,stream1);
    //     cudaMemcpyAsync(c_h+i+2*N,c_d,N*sizeof(int),cudaMemcpyDeviceToHost,stream2);
    //     cudaMemcpyAsync(c_h+i+3*N,c_d,N*sizeof(int),cudaMemcpyDeviceToHost,stream3);
    // } 
    // cudaStreamSynchronize(stream0);
    // cudaStreamSynchronize(stream1);
    // cudaStreamSynchronize(stream2);
    // cudaStreamSynchronize(stream3);
    // cudaEventRecord(stop,0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime,start,stop);
    // printf("time 4 stream=%fms\n",elapsedTime);

    //释放
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaFreeHost(a_h);
    cudaFreeHost(b_h);
    cudaFreeHost(c_h);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    return 0;
} 