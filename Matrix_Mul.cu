#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
using namespace std;
#define BLOCK_WIDTH 16
#define BLOCK_HIGH 16
#include <stdio.h>
//朴素矩阵乘法
__global__ void Mat_Mul(float *A,float*B,float*C,int WIDTH,int ComLong)
{
    //获得行、列
    int row=BLOCK_HIGH*blockIdx.y+threadIdx.y;
    int col=BLOCK_WIDTH*blockIdx.x+threadIdx.x;
    float Cvalue=0;
    //A中每一行的每一个元素*B中每一列的每一个元素的和
    for(int k=0;k<ComLong;k++)
    {
        Cvalue+=A[row*ComLong+k]*B[k*WIDTH+col];
    }
    C[row*WIDTH+col]=Cvalue;
}

//使用共享内存的矩阵乘法
__global__ void Mat_Mul_shard(float *A,float*B,float*C,int WIDTH,int ComLong)
{
    //定义共享内存大小
    __shared__ float Ads[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float Bds[BLOCK_WIDTH][BLOCK_WIDTH];
    //获得行、列
    int row=BLOCK_HIGH*blockIdx.y+threadIdx.y;
    int col=BLOCK_WIDTH*blockIdx.x+threadIdx.x;
    float Cvalue=0;
    //
    for(int m=0;m<(ComLong+BLOCK_WIDTH-1)/BLOCK_WIDTH;m++)
    {
        //把数据存入共享内存
        Ads[threadIdx.y][threadIdx.x]=A[row*ComLong+(m*BLOCK_WIDTH+threadIdx.x)];
        Bds[threadIdx.y][threadIdx.x]=B[(m*BLOCK_WIDTH+threadIdx.y)*WIDTH+col];
        __syncthreads();
        //计算当前共享block里的行*列
        for(int k=0;k<BLOCK_WIDTH;k++)
        {
        Cvalue+=Ads[threadIdx.y][k]*Bds[k][threadIdx.x];
        }
         __syncthreads();//这里不同步可能就有线程把后续block里的数据存到shared上，导致取的数据是错的 
    }
    C[row*WIDTH+col]=Cvalue;    
}

//使用有bank冲突的共享内存的矩阵乘法
__global__ void Mat_Mul_shard_bank(float *A,float*B,float*C,int WIDTH,int ComLong)
{
    //定义共享内存大小
    __shared__ float Ads[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float Bds[BLOCK_WIDTH][BLOCK_WIDTH];
    //获得行、列
    int row=BLOCK_HIGH*blockIdx.x+threadIdx.x;
    int col=BLOCK_WIDTH*blockIdx.y+threadIdx.y;
    float Cvalue=0;
    //
    for(int m=0;m<(ComLong+BLOCK_WIDTH-1)/BLOCK_WIDTH;m++)
    {
        //把数据存入共享内存
        Ads[threadIdx.x][threadIdx.y]=A[row*ComLong+(m*BLOCK_WIDTH+threadIdx.y)];
        Bds[threadIdx.x][threadIdx.y]=B[(m*BLOCK_WIDTH+threadIdx.x)*WIDTH+col];
        __syncthreads();
        //计算当前共享block里的行*列
        for(int k=0;k<BLOCK_WIDTH;k++)
        {
        Cvalue+=Ads[threadIdx.x][k]*Bds[k][threadIdx.y];
        }
         __syncthreads();//这里不同步可能就有线程把后续block里的数据存到shared上，导致取的数据是错的 
    }
    C[row*WIDTH+col]=Cvalue;    
}

// //test使用有bank冲突的共享内存的矩阵乘法
// __global__ void Mat_Mul_shard_bank(float *A,float*B,float*C,int WIDTH,int ComLong)
// {
//     //定义共享内存大小
//     __shared__ float Ads[BLOCK_WIDTH][BLOCK_WIDTH];
//     __shared__ float Bds[BLOCK_WIDTH][BLOCK_WIDTH];
//     //获得行、列
//     int row=BLOCK_HIGH*blockIdx.y+threadIdx.y;
//     int col=BLOCK_WIDTH*blockIdx.x+threadIdx.x;
//     float Cvalue=0;
//     //
//     for(int m=0;m<(ComLong+BLOCK_WIDTH-1)/BLOCK_WIDTH;m++)
//     {
//         //把数据存入共享内存
//         Ads[threadIdx.y][threadIdx.x]=A[row*ComLong+(m*BLOCK_WIDTH+threadIdx.x)];
//         Bds[threadIdx.y][threadIdx.x]=B[(m*BLOCK_WIDTH+threadIdx.y)*WIDTH+col];
//         __syncthreads();
//         //计算当前共享block里的行*列
//         for(int k=0;k<BLOCK_WIDTH;k++)
//         {
//         Cvalue+=Ads[threadIdx.y][k]*Bds[k][threadIdx.x];
//         }
//          __syncthreads();//这里不同步可能就有线程把后续block里的数据存到shared上，导致取的数据是错的 
//     }
//     C[row*WIDTH+col]=Cvalue;    
// }
// //解决了bank冲突的共享内存的矩阵乘法
__global__ void Mat_Mul_shard_bank_fix(float *A,float*B,float*C,int WIDTH,int ComLong)
{
    //定义共享内存大小
    __shared__ float Ads[BLOCK_WIDTH][BLOCK_WIDTH+1];
    __shared__ float Bds[BLOCK_WIDTH][BLOCK_WIDTH+1];
    //获得行、列
    int row=BLOCK_HIGH*blockIdx.x+threadIdx.x;
    int col=BLOCK_WIDTH*blockIdx.y+threadIdx.y;
    float Cvalue=0;
    //
    for(int m=0;m<(ComLong+BLOCK_WIDTH-1)/BLOCK_WIDTH;m++)
    {
        //把数据存入共享内存
        Ads[threadIdx.x][threadIdx.y]=A[row*ComLong+(m*BLOCK_WIDTH+threadIdx.y)];
        Bds[threadIdx.x][threadIdx.y]=B[(m*BLOCK_WIDTH+threadIdx.x)*WIDTH+col];
        __syncthreads();
        //计算当前共享block里的行*列
        for(int k=0;k<BLOCK_WIDTH;k++)
        {
        Cvalue+=Ads[threadIdx.x][k]*Bds[k][threadIdx.y];
        }
         __syncthreads();//这里不同步可能就有线程把后续block里的数据存到shared上，导致取的数据是错的 
    }
    C[row*WIDTH+col]=Cvalue;    
}
int main()
{
    const int WIDTH=3096;
    const int HIGH=3096;
    const int Xab=3096;
    //printf("WIDTH=%d\n",WIDTH); 
   // 定义cpu上的矩阵
    float (*array1_h)[Xab] = (float (*)[Xab])malloc(HIGH * sizeof(float[Xab]));
    // float *array1_h = malloc(WIDTH * sizeof(float[WIDTH]));
    // float (*array1_h)[B] = (float (*)[B])malloc(A * sizeof(float[B]));
    // array1_h[B][A]
    float (*array2_h)[WIDTH] = (float (*)[WIDTH])malloc(Xab* sizeof(float[WIDTH]));
    float (*result_h)[WIDTH] = (float(*)[WIDTH])malloc(HIGH*sizeof(float[WIDTH]));
    float (*test_h)[WIDTH] = (float(*)[WIDTH])malloc(HIGH*sizeof(float[WIDTH]));//测试矩阵
      //初始化矩阵
    int i=0,j=0;
    for(i=0;i<HIGH;i++)//array1_h初始化
    {
        for(j=0;j<Xab;j++)
        {
            array1_h[i][j]=i;
        }
    }
     for(i=0;i<Xab;i++)//array2_h初始化
    {
        for(j=0;j<WIDTH;j++)
        {
            array2_h[i][j]=j;
        }
    }
    //printf("array1_h=%f\n",array1_h[1][10]);
    //printf("array2_h=%f\n",array2_h[10][1]);
    //cpu运算矩阵乘法
    for(i=0;i<HIGH;i++)
    {
        for(j=0;j<WIDTH;j++)
        {
            for(int k=0;k<Xab;k++)
            {
            test_h[i][j]+=array1_h[i][k]*array2_h[k][j];
            }
        }
    }
    //printf("test_h[0][1]=%f\n",test_h[1][1]);
    // printf("test_h[1][1]=%f\n",test_h[1][1]);
    // printf("test_h[2][1]=%f\n",test_h[2][1]);
    // printf("test_h[2][2]=%f\n",test_h[2][2]);
    //定义时间事件
    cudaEvent_t start,stop1,stop2;
    float elapsedTime,elapsedTimecpy,timeall;
    cudaEventCreate(&start);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);
    //定义GPU上矩阵
    int size=WIDTH*HIGH;
    int bytes=size*sizeof(int);
    //printf("size=%d,bytes=%d\n",size,bytes);
    float *array1_d;
    float *array2_d;
    float *result_d;
    cudaMalloc((void**)&array1_d,HIGH*Xab*sizeof(float));
    cudaMalloc((void**)&array2_d,Xab*WIDTH*sizeof(float));
    cudaMalloc((void**)&result_d,bytes);
    

    //cpu传GPU
    cudaEventRecord(start);
    cudaMemcpy(array1_d,array1_h,HIGH*Xab*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(array2_d,array2_h,Xab*WIDTH*sizeof(float),cudaMemcpyHostToDevice);
    
    //定义kernel函数的执行设置
    dim3 blocksize(BLOCK_WIDTH,BLOCK_HIGH,1);
    dim3 gridsize((WIDTH+BLOCK_WIDTH-1)/BLOCK_WIDTH,(WIDTH+BLOCK_HIGH-1)/BLOCK_HIGH,1);
    printf("blocksize.x=%d,blocksize.y=%d,blocksize.z=%d\n",blocksize.x,blocksize.y,blocksize.z);
    printf("gridsize.x=%d,gridsize.y=%d,gridsize.z=%d\n",gridsize.x,gridsize.y,gridsize.z);
    cudaEventRecord(stop1);
    cudaEventElapsedTime(&elapsedTimecpy,start,stop1);
    printf("cpy time=%d\n",elapsedTimecpy);
   //热身
    Mat_Mul<<<gridsize,blocksize>>>(array1_d,array2_d,result_d,WIDTH,Xab);
   // 执行kernel函数

    //朴素矩阵乘法
    cudaEventRecord(start,0);
    Mat_Mul<<<gridsize,blocksize>>>(array1_d,array2_d,result_d,WIDTH,Xab);
    
        //GPU->CPU
    cudaMemcpy(result_h,result_d,bytes,cudaMemcpyDeviceToHost);
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&elapsedTime,start,stop1);
    timeall=elapsedTime+elapsedTimecpy;
    printf("Mat_MUl runtime=%f,Mat_MUl alltime=%f\n",elapsedTime,timeall);
        //验证
    // printf("result_h[1][1]%f\n",result_h[1][1]);
    // printf("result_h[2][1]%f\n",result_h[2][1]);
    // printf("result_h[2][2]%f\n",result_h[2][2]);
    for(i=0;i<WIDTH;i++)
    {
        
        for(j=0;j<WIDTH;j++)
        {
            if(fabs(test_h[i][j]-result_h[i][j])!=0)
            {
                printf("Result verification failed at element%d\n",i*WIDTH+j);
                exit(EXIT_FAILURE);
            }
        }
    }
    printf("testpass\n");

   //使用了共享内存的矩阵乘法
    cudaEventRecord(start,0);
    Mat_Mul_shard<<<gridsize,blocksize>>>(array1_d,array2_d,result_d,WIDTH,Xab);
        //GPU->CPU
    cudaMemcpy(result_h,result_d,bytes,cudaMemcpyDeviceToHost);
    cudaEventRecord(stop2,0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime,start,stop2);
    timeall=elapsedTime+elapsedTimecpy;
    printf("Mat_MUl_shared runtime=%f,Mat_MUl_shared alltime=%f\n",elapsedTime,timeall);
        //验证
    // printf("result_h[1][1]%f\n",result_h[1][1]);
    // printf("result_h[2][1]%f\n",result_h[2][1]);
    // printf("result_h[2][2]%f\n",result_h[2][2]);
    for(i=0;i<WIDTH;i++)
    {
        for(j=0;j<WIDTH;j++)
        {
            if(fabs(test_h[i][j]-result_h[i][j])!=0)
            {
                printf("Result verification failed at element%d\n",i*WIDTH+j);
                exit(EXIT_FAILURE);
            }
        }
    }
    printf("testpass\n");

     // //使用bank冲突的共享内存的矩阵乘法
    cudaEventRecord(start,0);
    Mat_Mul_shard_bank<<<gridsize,blocksize>>>(array1_d,array2_d,result_d,WIDTH,Xab);
        //GPU->CPU
    cudaMemcpy(result_h,result_d,bytes,cudaMemcpyDeviceToHost);
    cudaEventRecord(stop2,0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime,start,stop2);
    timeall=elapsedTime+elapsedTimecpy;
    printf("Mat_MUl_shared_bank runtime=%f,Mat_MUl_shared_bank alltime=%f\n",elapsedTime,timeall);
    // //   验证
    // printf("result_h[1][1]%f\n",result_h[1][1]);
    // printf("result_h[2][1]%f\n",result_h[2][1]);
    // printf("result_h[2][2]%f\n",result_h[2][2]);
    for(i=0;i<WIDTH;i++)
    {
        for(j=0;j<WIDTH;j++)
        {
            if(fabs(test_h[i][j]-result_h[i][j])!=0)
            {
                printf("Result verification failed at element%d\n",i*WIDTH+j);
                exit(EXIT_FAILURE);
            }
        }
    }
    printf("testpass\n");

    //解决bank冲突的共享内存的矩阵乘法
    cudaEventRecord(start,0);
    Mat_Mul_shard_bank_fix<<<gridsize,blocksize>>>(array1_d,array2_d,result_d,WIDTH,Xab);
            //GPU->CPU
    cudaMemcpy(result_h,result_d,bytes,cudaMemcpyDeviceToHost);
    cudaEventRecord(stop2,0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime,start,stop2);
    timeall=elapsedTime+elapsedTimecpy;
    printf("Mat_MUl_shared_bank_fix runtime=%f,Mat_MUl_shared_bank_fix alltime=%f\n",elapsedTime,timeall);
    // //验证
    // printf("result_h[1][1]%f\n",result_h[1][1]);
    // printf("result_h[2][1]%f\n",result_h[2][1]);
    // printf("result_h[2][2]%f\n",result_h[2][2]);
    for(i=0;i<WIDTH;i++)
    {
        for(j=0;j<WIDTH;j++)
        {
            if(fabs(test_h[i][j]-result_h[i][j])!=0)
            {
                printf("Result verification failed at element%d\n",i*WIDTH+j);
                exit(EXIT_FAILURE);
            }
        }
    }
    printf("testpass\n");
    //使用共享内存+流的矩阵乘法
    //  //为了使用流申请的
    // float *array1_d1;
    // float *array2_d1;
    // float *result_d1;
    // float *array1_d2;
    // float *array2_d2;
    // float *result_d2;
    // cudaHostAlloc((void**)&array1_d1,HIGH*Xab*sizeof(float)/2,cudaHostAllocDefault);
    // cudaHostAlloc((void**)&array2_d1,Xab*WIDTH*sizeof(float)/2,cudaHostAllocDefault);
    // cudaHostAlloc((void**)&result_d1,bytes/2,cudaHostAllocDefault);
    // cudaHostAlloc((void**)&array1_d2,HIGH*Xab*sizeof(float)/2,cudaHostAllocDefault);
    // cudaHostAlloc((void**)&array2_d2,Xab*WIDTH*sizeof(float)/2,cudaHostAllocDefault);
    // cudaHostAlloc((void**)&result_d2,bytes/2,cudaHostAllocDefault);
    // cudaStream_t stream0,stream1;
    // cudaStreamCreate(&stream0);
    // cudaStreamCreate(&stream1);
    // //计算开始
    // printf("1\n");
    // //cudaEventRecord(start,0);
    // cudaMemcpyAsync(array1_d1,array1_h,HIGH*Xab*sizeof(float)/2,cudaMemcpyHostToDevice,stream0);
    // cudaMemcpyAsync(array1_d2,array1_h+256,HIGH*Xab*sizeof(float)/2,cudaMemcpyHostToDevice,stream1);
    // cudaMemcpyAsync(array2_d1,array2_h,Xab*WIDTH*sizeof(float)/2,cudaMemcpyHostToDevice,stream0);
    // //cudaMemcpyAsync(array2_d2,array2_h+(Xab*WIDTH)/2,Xab*WIDTH*sizeof(float)/2,cudaMemcpyHostToDevice,stream1);
    // dim3 blocksize_1(BLOCK_WIDTH,BLOCK_HIGH,1);
    // dim3 gridsize_1(((WIDTH+BLOCK_WIDTH-1)/BLOCK_WIDTH)/2,((WIDTH+BLOCK_HIGH-1)/BLOCK_HIGH,1)/2);
    // printf("2\n");
    // Mat_Mul_shard<<<gridsize_1,blocksize_1,0,stream0>>>(array1_d1,array2_d1,result_d1,WIDTH,Xab);
    // //Mat_Mul_shard<<<gridsize_1,blocksize_1,0,stream1>>>(array1_d2,array2_d2,result_d2,WIDTH,Xab);
    // printf("3\n");
    // cudaMemcpyAsync(result_h,result_d1,bytes/2,cudaMemcpyDeviceToHost,stream0);
    // //cudaMemcpyAsync(result_h+WIDTH*HIGH/2,result_d2,bytes/2,cudaMemcpyDeviceToHost,stream1);
    // printf("4\n");
    //  // //验证
    // // printf("result_h[1][1]%f\n",result_h[1][1]);
    // // printf("result_h[2][1]%f\n",result_h[2][1]);
    // // printf("result_h[2][2]%f\n",result_h[2][2]);
    // for(i=0;i<WIDTH;i++)
    // {
    //     for(j=0;j<WIDTH;j++)
    //     {
    //         if(fabs(test_h[i][j]-result_h[i][j])!=0)
    //         {
    //             printf("Result verification failed at element%d\n",i*WIDTH+j);
    //             exit(EXIT_FAILURE);
    //         }
    //     }
    // }
    // printf("testpass\n");
    // cudaStreamSynchronize(stream0);
    // cudaStreamSynchronize(stream1);
    // //释放空间
    // cudaFree(array1_d);
    // cudaFree(array2_d);
    // cudaFree(result_d);
    // cudaFreeHost(array1_h);
    // cudaFreeHost(array2_h);
    // cudaFreeHost(result_h);
    
    // cudaFreeHost(array1_d1);
    // cudaFreeHost(array2_d1);
    // cudaFreeHost(result_d1);
    // cudaFreeHost(array1_d2);
    // cudaFreeHost(array2_d2);
    // cudaFreeHost(result_d2);
    // cudaStreamDestroy(stream0);
    // cudaStreamDestroy(stream1);
   
    return 0;
}