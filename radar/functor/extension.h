#pragma once
#include "../core/sequence.h"

namespace radar::functor
{
    struct sgn
    {
        template <typename T>
        __host__ __device__ inline
        T operator()(T x) const
        {
            return x == T(0) ? T(0) : (x > T(0) ? T(1) : T(-1));
        }
    };

    struct pow2
    {
        template <typename T>
        __host__ __device__ inline
        T operator()(T x) const
        {
            return x * x;
        }

        template<typename Arg, typename Mult>
        auto accumulate_grad(const engine::expr<Arg> &arg, const engine::expr<Mult> &mult) const
        {
            return constant<typename Arg::value_type>(2) * arg * mult;
        }
    };
}
