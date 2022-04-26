#pragma once

namespace radar::functor
{
    struct sigmoid
    {
        template <typename T>
        __host__ __device__ inline
        T operator()(T x) const
        {
            return T(1) / (T(1) + ::exp(-x));
        }
    };
}
