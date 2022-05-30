#pragma once
#include <device_launch_parameters.h>

namespace radann::cuda::kernel
{
    template <typename T, typename Expr>
    __global__ void assign(T*, size_t, Expr);

    template <typename T>
    __global__ void reverse_grad(T*, const T*, const T*, size_t, size_t);
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
    __global__ void reverse_grad(T *input_grad,
                                 const T *mult, const T *output_grad,
                                 size_t input_size, size_t output_size)
    {
        for (auto i = blockIdx.x * blockDim.x + threadIdx.x;
                i < input_size;
                i += blockDim.x * gridDim.x)
            input_grad[i] += mult[i] * output_grad[i % output_size];
    }
}
