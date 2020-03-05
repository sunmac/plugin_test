#include "cublas_v2.h"
#include "plugin.h"
#include <cassert>
#include <cstdio>

pluginStatus_t Sliceinference(cudaStream_t stream,const int x1,const int y1,const int z1,const int x2,const int y2,
    const int z2,const float* input_data,const float* input_slice,float* output,int batchSize);


pluginStatus_t Sliceinference(cudaStream_t stream,const int x1,const int y1,const int z1,const int x2,const int y2,
    const int z2,  const half* input_data,  const half* input_slice,half* output,int batchSize);