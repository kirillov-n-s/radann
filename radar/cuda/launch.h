#pragma once
#include "prop_context.h"

namespace radar::cuda
{
    template<typename Kernel>
    void get_launch_parameters(const Kernel&, size_t, int&, int&);
}

namespace radar::cuda
{
    template<typename Kernel>
    void get_launch_parameters(const Kernel &kernel, size_t size, int& block_dim, int& grid_dim)
    {
        cudaOccupancyMaxPotentialBlockSize(
                &grid_dim,
                &block_dim,
                kernel,
                0,
                size);
        int blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm,
                kernel,
                block_dim,
                0);
        grid_dim = std::min<int>((size - 1) / block_dim + 1, blocks_per_sm * get_prop()->multiProcessorCount);
    }
}
