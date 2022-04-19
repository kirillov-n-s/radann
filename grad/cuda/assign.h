#pragma once
#include <string>
#include <cuda_runtime.h>
#include "kernel/assign.h"
#include "prop.h"

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
        int block_dim, grid_dim;
        cudaOccupancyMaxPotentialBlockSize(
                &grid_dim,
                &block_dim,
                kernel::assign<T, Expr>,
                0,
                size);
        int blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm,
                kernel::assign<T, Expr>,
                block_dim,
                0);
        grid_dim = std::min<int>((size - 1) / block_dim + 1, blocks_per_sm * get_prop()->multiProcessorCount);

        kernel::assign<<<grid_dim, block_dim>>>(data, size, expr);
        auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            throw std::runtime_error("Assign kernel failed. CUDA error status " + std::to_string(status));
    }
}
