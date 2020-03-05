#include "NvInfer.h"
#include "plugin.h"
// #include "common.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <vector>
#include"cuda_fp16.h"
// template<typename T>
// __global__ void Slice (const T* input_data,const T* input_slice,T* output)
// {
//     int i= blockIdx.x*gridDim.y+blockIdx.y*blockDim.y + threadIdx.x;
//     // printf("index %d input_data %f input_slice %f\n",i,input_data[i],input_slice[i]);
//     // printf("blockDim.x  %d blockDim %d input_slice %d\n",blockDim.x ,gridDim.y,gridDim.x);
//     // output[i]=input_data[i];
//     // const T in=(input_data[i]+input_slice[i]);
//     output[i]=(input_data[i]);
//     // *(output+i)=*(input_data+i);
//     //*(output+i)=*(input_data+i)+*(input_slice+i);
// }
// __global__ void Slice (const half* input_data,const half* input_slice,half* output)
// {
//     int i= blockIdx.x*gridDim.y+blockIdx.y*blockDim.y + threadIdx.x;
//     output[i]=__hadd(input_data[i],input_slice[i]);
//     // *(output+i)=*(input_data+i);
//     //*(output+i)=*(input_data+i)+*(input_slice+i);
// }
// pluginStatus_t Sliceinference(cudaStream_t stream,const int x1,const int y1,const int z1,const int x2,const int y2,
//     const int z2,const float* input_data,const float* input_slice,float* output)
// {
//     dim3 dimBlock(x1,y1);
//     dim3 dimthread(z1);
//     std::cout<<x1<<" "<<y1<<" "<<z1<<std::endl;
//     Slice<float><<<dimBlock,dimthread,0,stream>>>(input_data, input_slice, output);
//     return STATUS_SUCCESS;
// }


// pluginStatus_t Sliceinference(cudaStream_t stream,const int x1,const int y1,const int z1,const int x2,const int y2,
//     const int z2,  const half* input_data,  const half* input_slice,half* output)
// {
//     dim3 dimBlock(x1,y1);
//     dim3 dimthread(z1);
//     // std::cout<<x1<<" "<<y1<<" "<<z1<<std::endl;
//     // const half2* input_data_ = reinterpret_cast<const half2*>(input_data);
//     // const half2* input_slice_ = reinterpret_cast<const half2*>(input_slice);
//     // half2* output_ = reinterpret_cast<half2*>(output);

//     Slice<<<dimBlock,dimthread,0,stream>>>(input_data, input_slice, output);
//     return STATUS_SUCCESS;
// }

template<typename T>
__global__ void Slice (const T* input_data,const T* input_slice,T* output)
{
    int i= blockIdx.x*blockDim.y*blockDim.x+threadIdx.x*blockDim.y + threadIdx.y;
    // printf("%d %f %f %d %d %d\n",i,input_slice[i],input_data[threadIdx.x*blockDim.y + threadIdx.y],blockIdx.x,threadIdx.x,blockIdx.y);
    output[i]=input_slice[i]+input_data[threadIdx.x*blockDim.y + threadIdx.y];

}
__global__ void Slice (const half* input_data,const half* input_slice,half* output)
{
    int i= blockIdx.x*blockDim.y*blockDim.x+threadIdx.x*blockDim.y + threadIdx.y;
    output[i]=__hadd(input_data[threadIdx.x*blockDim.y + threadIdx.y],input_slice[i]);
}
pluginStatus_t Sliceinference(cudaStream_t stream,const int x1,const int y1,const int z1,const int x2,const int y2,
    const int z2,const float* input_data,const float* input_slice,float* output,int batchsize)
{
    dim3 dimBlock(y2,z2);
    dim3 dimgrid(batchsize);
    std::cout<<"cuda"<<y2<<" "<<z2<<" "<<batchsize<<std::endl;
    Slice<float><<<dimgrid,dimBlock,0,stream>>>(input_data, input_slice, output);
    return STATUS_SUCCESS;
}


pluginStatus_t Sliceinference(cudaStream_t stream,const int x1,const int y1,const int z1,const int x2,const int y2,
    const int z2,  const half* input_data,  const half* input_slice,half* output,int batchsize)
{
    dim3 dimBlock(y1,z1);
    dim3 dimgrid(batchsize);
    Slice<<<dimgrid,dimBlock,0,stream>>>(input_data, input_slice, output);
    return STATUS_SUCCESS;
}