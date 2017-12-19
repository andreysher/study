#include "kernel_cuda.h"

__global__
void matrixMulShared(
        const DT *mat1,
        const DT *mat2,
        const DT alpha,
        DT *out)
{
    unsigned int width = gridDim.x * blockDim.x;

    const unsigned int blockSize = 16;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ DT As[blockSize][blockSize];
    __shared__ DT Bs[blockSize][blockSize];

    int aBegin = width * blockSize * by;
    int aEnd = aBegin + width - 1;
    int aStep = blockSize;
    int bBegin = blockSize * bx;
    int bStep = blockSize * width;

    DT Csub = 0;

    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
        As[ty][tx] = mat1[a + width * ty + tx];
        Bs[ty][tx] = mat2[b + width * ty + tx];

        __syncthreads();

        for (int k = 0; k < blockSize; ++k)
            Csub += As[ty][k] * Bs[k][tx] * alpha;

        __syncthreads();
    }

    int c = width * blockSize * by + blockSize * bx;
    out[c + width * ty + tx] = Csub;
}

extern "C"
void sgemm(dim3 gridDim, dim3 blockDim, mydata_t *params, DT alpha)
{
		matrixMulShared << <gridDim, blockDim >> >(params->devData[0], 
    			params->devData[1], alpha, params->devData[2]);
}
