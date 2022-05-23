#pragma once
#include <device_launch_parameters.h>

namespace radann::cuda::kernel
{
    template <typename T, typename Expr>
    __global__ void assign(T*, size_t, Expr);

    template <typename T>
    __global__ void fma(T*, const T*, const T*, size_t);
}

namespace radann::cuda::kernel
{
    template <typename T, typename Expr>
    __global__ void assign(T *data, size_t size, Expr expr)
    {
        for (auto i = blockIdx.x * blockDim.x + threadIdx.x;
                i < size;
                i += blockDim.x * gridDim.x)
            data[i] = expr[i];
    }

    template <typename T>
    __global__ void fma(T *lvalue, const T *mult, const T *rvalue, size_t size)
    {
        for (auto i = blockIdx.x * blockDim.x + threadIdx.x;
                i < size;
                i += blockDim.x * gridDim.x)
            lvalue[i] += mult[i] * rvalue[i];
    }
}
