#include "cublas_v2.h"
#include "plugin.h"
#include <cassert>
#include <cstdio>

void resizeNearest(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scale, int2 osize, float const* idata,
    int istride, int ibatchstride, float* odata, int ostride, int obatchstride);