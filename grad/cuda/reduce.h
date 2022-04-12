#pragma once
#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>

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
        auto c = thrust::make_constant_iterator(1);
        auto d = thrust::make_discard_iterator();
        thrust::reduce_by_key(thrust::device, c, c + size, data, d, res, thrust::equal_to<T>{}, op);
    }
}
