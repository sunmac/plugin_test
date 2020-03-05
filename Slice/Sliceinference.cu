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
template<typename T>
__global__ void Slice (const T* input_data,const T* input_slice,T* output)
{
    int i= blockIdx.x*blockDim.y*blockDim.x+threadIdx.x*blockDim.y + threadIdx.y;
    printf("%d %f %f %d %d %d\n",i,input_slice[i],input_data[threadIdx.x*blockDim.y + threadIdx.y],blockIdx.x,threadIdx.x,blockIdx.y);
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
    std::cout<<x1<<" "<<y1<<" "<<z1<<" "<<batchsize<<std::endl;
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