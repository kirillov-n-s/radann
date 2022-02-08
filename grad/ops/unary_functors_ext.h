#pragma once

namespace grad::ops
{
    struct sgn
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x) const
        {
            return x / ::fabs(x);
        }
    };
}
