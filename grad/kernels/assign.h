#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace grad::par::kernels
{
    template <typename T, typename Expr>
    __global__ void assign(T*, size_t, Expr);
}

namespace grad::par::kernels
{
    template <typename T, typename Expr>
    __global__ void assign(T *data, size_t size, Expr expr)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size)
            data[i] = expr[i];
    }
}
