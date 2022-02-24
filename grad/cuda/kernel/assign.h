#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace grad::cuda::kernel
{
    template <typename T, typename Expr>
    __global__ void assign(T*, size_t, Expr);
}

namespace grad::cuda::kernel
{
    template <typename T, typename Expr>
    __global__ void assign(T *data, size_t size, Expr expr)
    {
        for (auto i = blockIdx.x * blockDim.x + threadIdx.x;
                i < size;
                i += blockDim.x * gridDim.x)
            data[i] = expr[i];
    }
}
