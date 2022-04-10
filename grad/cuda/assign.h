#pragma once
#include <string>
#include "kernel/assign.h"
#include "launch.h"

namespace grad::cuda
{
    template <typename T, typename Expr>
    void assign(T*, size_t, const Expr&);
}

namespace grad::cuda
{
    template<typename T, typename Expr>
    void assign(T *data, size_t size, const Expr &expr)
    {
        size_t block_dim, grid_dim;
        get_launch_config(kernel::assign<T, Expr>, size, 0, block_dim, grid_dim);
        kernel::assign<<<grid_dim, block_dim>>>(data, size, expr);
        auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            throw std::runtime_error("Assign kernel failed. CUDA error status " + std::to_string(status));
    }
}
