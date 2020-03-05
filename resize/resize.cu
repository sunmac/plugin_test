
#include <cuda_runtime.h>
__global__ void resize_nearest_kernel_2d(int nbatch, float scale, int2 osize, float const* idata, int istride,
    int ibatchstride, float* odata, int ostride, int obatchstride)
{

    int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    int y0 = threadIdx.y + blockIdx.y * blockDim.y;
    int z0 = blockIdx.z;
    for (int batch = z0; batch < nbatch; batch += gridDim.z)
    {
        for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y)
        {
            for (int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x)
            {
                int ix = int(ox / scale);
                int iy = int(oy / scale);
                odata[batch * obatchstride + oy * ostride + ox] = idata[batch * ibatchstride + iy * istride + ix];
            }
        }
    }
}

void resizeNearest(dim3 grid, dim3 block, cudaStream_t stream, int nbatch, float scale, int2 osize, float const* idata,
    int istride, int ibatchstride, float* odata, int ostride, int obatchstride)
{

    resize_nearest_kernel_2d<<<grid, block, 0, stream>>>(
        nbatch, scale, osize, idata, istride, ibatchstride, odata, ostride, obatchstride);
}