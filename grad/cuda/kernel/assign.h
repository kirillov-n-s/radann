#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace grad::cuda
{
    enum class assign_t
    {
        reg, add, sub, mul, div
    };
}

namespace grad::cuda::kernel
{
    template <assign_t A, typename T, typename Expr>
    __global__ void assign(T*, size_t, Expr);
}

namespace grad::cuda::kernel
{
    template <assign_t A, typename T, typename Expr>
    __global__ void assign(T *data, size_t size, Expr expr)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size)
        {
            if constexpr(A == assign_t::reg)
                data[i] = expr[i];
            else if constexpr(A == assign_t::add)
                data[i] += expr[i];
            else if constexpr(A == assign_t::sub)
                data[i] -= expr[i];
            else if constexpr(A == assign_t::mul)
                data[i] *= expr[i];
            else
                data[i] /= expr[i];
        }
    }
}
