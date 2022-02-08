#pragma once
#include "kernels/assign.h"

namespace grad::par
{
    template <typename T, typename Expr>
    void assign(T*, size_t, const Expr&);
}

namespace grad::par
{
    template<typename T, typename Expr>
    void assign(T *data, size_t size, const Expr &expr)
    {
        auto num_threads = 256;
        kernels::assign<<<(size - 1) / num_threads + 1, num_threads>>>(data, size, expr);
        //cudaMemPrefetchAsync(data, size, 0);
        auto status = cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::runtime_error("");
    }
}
