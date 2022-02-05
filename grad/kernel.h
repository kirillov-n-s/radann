#pragma once
#include "thrust/copy.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/for_each.h"

namespace grad::kernel
{
    template <typename InputIterator, typename T>
    void copy(InputIterator, InputIterator, T*);

    template <typename T, typename Expr>
    void assign(T*, size_t, const Expr&);
}

namespace grad::kernel
{
    template <typename InputIterator, typename T>
    void copy(InputIterator first, InputIterator last, T *dst)
    {
        thrust::copy(first, last, dst);
    }

    template <typename T, typename Expr>
    void assign(T *dst, size_t size, const Expr& expr)
    {
        auto begin = thrust::make_counting_iterator(0);
        auto end = begin + size;
        thrust::for_each(thrust::device, begin, end,
            [=] __device__ (size_t i)
            {
                dst[i] = expr[i];
            });
    }
}
