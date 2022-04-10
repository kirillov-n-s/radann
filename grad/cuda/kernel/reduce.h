#pragma once
#include "device_launch_parameters.h"
#include "shared.h"

namespace grad::cuda::kernel
{
    template <typename T, typename Op>
    __device__ void warp_reduce(volatile T*, size_t, Op);

    template <typename T, typename Op>
    __global__ void reduce(const T*, T*, size_t, Op);
}

namespace grad::cuda::kernel
{
    template <typename T, typename Op>
    __device__ void warp_reduce(volatile T *shared, size_t idx, Op op)
    {
        shared[idx] = op(shared[idx], shared[idx + 32]);
        shared[idx] = op(shared[idx], shared[idx + 16]);
        shared[idx] = op(shared[idx], shared[idx + 8]);
        shared[idx] = op(shared[idx], shared[idx + 4]);
        shared[idx] = op(shared[idx], shared[idx + 2]);
        shared[idx] = op(shared[idx], shared[idx + 1]);
    }

    template <typename T, typename Op>
    __global__ void reduce(const T* data, T* res, size_t size, Op op)
    {
        auto shared = extern_shared_memory<T>();

        auto block_dim = blockDim.x;
        auto grid_stride = block_dim * 2 * gridDim.x;
        auto local_idx = threadIdx.x;
        auto global_idx = blockIdx.x * block_dim * 2 + threadIdx.x;

        if (global_idx >= size)
            return;

        shared[local_idx] = op(data[global_idx], data[global_idx + block_dim]);
        for (global_idx += grid_stride; global_idx < size; global_idx += grid_stride)
            shared[local_idx] = op(shared[local_idx], op(data[global_idx], data[global_idx + block_dim]));
        __syncthreads();

        for (size_t offset = block_dim >> 1; offset > 32; offset >>= 1)
        {
            if (local_idx < offset)
                shared[local_idx] = op(shared[local_idx], shared[local_idx + offset]);
            __syncthreads();
        }

        if (local_idx < 32)
            warp_reduce(shared, local_idx, op);

        if (local_idx == 0)
            res[blockIdx.x] = shared[0];
    }
}
