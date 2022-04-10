#pragma once
#include "kernel/reduce.h"
#include "device_buffer.h"
#include "../util/util.h"

namespace grad::cuda
{
    template <typename T, typename Op>
    void reduce(const T*, T*, size_t, Op);
}

namespace grad::cuda
{
    template <typename T, typename Op>
    void reduce(const T *data, T *res, size_t size, Op op)
    {
        size_t block_dim, grid_dim;
        get_launch_config(kernel::reduce<T, Op>, util::half_ceil(size), [](auto x) { return x; }, block_dim, grid_dim);

        device_buffer<T> inter(grid_dim);
        auto* inter_data = inter.data();

        do
        {
            kernel::reduce<<<grid_dim, block_dim, block_dim * sizeof(T)>>>
                (data, inter_data, size, op);
            auto status = cudaDeviceSynchronize();
            if (status != cudaSuccess)
                throw std::runtime_error("Reduce kernel failed. CUDA error status " + std::to_string(status));

            if (grid_dim < block_dim)
                break;
            grid_dim = util::div_ceil(grid_dim, block_dim);
            data = inter_data;
        }
        while (true);

        //sequentially reduce block results when grid_dim < block_dim

        /*do
         {
            kernel::reduce<<<grid_dim, block_dim, block_dim * sizeof(T)>>>
                (data, inter.data(), size, op);
            auto status = cudaDeviceSynchronize();
            if (status != cudaSuccess)
                throw std::runtime_error("Reduce kernel failed. CUDA error status " + std::to_string(status));

            if (grid_dim == 1)
                break;
            grid_dim = (size_t)::ceil((float)grid_dim / block_dim);
        }
        while (true);*/

        inter.copy_to(res, 1);
    }
}
