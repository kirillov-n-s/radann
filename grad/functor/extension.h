#pragma once

namespace grad::functor
{
    struct sgn
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x) const
        {
            return x / ::fabs(x);
        }
    };

    struct pow2
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x) const
        {
            return ::pow(x, T(2));
        }
    };
}
