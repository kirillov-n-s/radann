#pragma once

namespace grad::functor
{
    struct sigmoid
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x) const
        {
            return 1 / (1 + ::exp(-x));
        }
    };
}
