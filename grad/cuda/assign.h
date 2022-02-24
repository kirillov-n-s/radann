#pragma once
#include <string>
#include "kernel/assign.h"

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
        int block_dim, min_grid_dim;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_dim,
                &block_dim,
                kernel::assign<T, Expr>,
                0,
                size);
        auto grid_dim = (size - 1) / block_dim + 1;

        int device;
        cudaGetDevice(&device);
        cudaMemPrefetchAsync(data, size, device);

        kernel::assign<<<grid_dim, block_dim>>>(data, size, expr);
        auto status = cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::runtime_error("Assign kernel failed. CUDA error status " + std::to_string(status));
    }
}
