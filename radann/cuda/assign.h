#pragma once
#include <string>
#include <cuda_runtime.h>
#include "kernel/assign.h"
#include "launch.h"

namespace radann::cuda
{
    template <typename T, typename Expr>
    void assign(T*, size_t, const Expr&);

    template <typename T>
    void fma(T*, const T*, const T*, size_t);
}

namespace radann::cuda
{
    template<typename T, typename Expr>
    void assign(T *data, size_t size, const Expr &expr)
    {
        int block_dim, grid_dim;
        get_launch_parameters(kernel::assign<T, Expr>, size, block_dim, grid_dim);
        kernel::assign<<<grid_dim, block_dim>>>(data, size, expr);
        auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            throw std::runtime_error("Assign kernel failed. CUDA error status " + std::to_string(status));
    }

    template <typename T>
    void fma(T *lvalue, const T *mult, const T *rvalue, size_t size)
    {
        int block_dim, grid_dim;
        get_launch_parameters(kernel::fma<T>, size, block_dim, grid_dim);
        kernel::fma<<<grid_dim, block_dim>>>(lvalue, mult, rvalue, size);
        auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            throw std::runtime_error("FMA kernel failed. CUDA error status " + std::to_string(status));
    }
}
