#pragma once
#include <string>
#include "kernel/assign.h"

namespace grad::cuda
{
    template <assign_t A, typename T, typename Expr>
    void assign(T*, size_t, const Expr&);
}

namespace grad::cuda
{
    template<assign_t A, typename T, typename Expr>
    void assign(T *data, size_t size, const Expr &expr)
    {
        auto num_threads = 256;
        auto num_blocks = (size - 1) / num_threads + 1;
        kernel::assign<A><<<num_blocks, num_threads>>>(data, size, expr);
        auto status = cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::runtime_error("Assign kernel failed. CUDA error status " + std::to_string(status));
    }
}
