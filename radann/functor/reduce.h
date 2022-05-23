#pragma once
#include "../cuda/reduce.h"
#include "../cuda/cublas.h"
#include "../core/array.h"
#include "binary.h"

namespace radann::functor
{
    struct sum
    {
        static constexpr bool requires_validation = false;

        template <size_t N, bool AD, typename T>
        auto operator()(const array<N, AD, T> &x) const
        {
            auto res = make_array<AD, T>(radann::make_shape());
            cuda::reduce(x.data(), res.data(), x.size(), add{});
            return res;
        }

        template<typename Arg, typename Mult>
        auto accumulate_grad(const engine::expr<Arg> &arg, const engine::expr<Mult> &mult) const
        {
            return ;
        }
    };

    struct prod
    {
        static constexpr bool requires_validation = false;

        template <size_t N, bool AD, typename T>
        auto operator()(const array<N, AD, T> &x) const
        {
            auto res = make_array<AD, T>(radann::make_shape());
            cuda::reduce(x.data(), res.data(), x.size(), mul{});
            return res;
        }
    };

    struct maxval
    {
        static constexpr bool requires_validation = false;

        template <size_t N, bool AD, typename T>
        auto operator()(const array<N, AD, T> &x) const
        {
            auto res = make_array<AD, T>(radann::make_shape());
            cuda::reduce(x.data(), res.data(), x.size(), max{});
            return res;
        }
    };

    struct minval
    {
        static constexpr bool requires_validation = false;

        template <size_t N, bool AD, typename T>
        auto operator()(const array<N, AD, T> &x) const
        {
            auto res = make_array<AD, T>(radann::make_shape());
            cuda::reduce(x.data(), res.data(), x.size(), min{});
            return res;
        }
    };

    struct norm2
    {
        static constexpr bool requires_validation = false;

        template <size_t N, bool AD, typename T>
        auto operator()(const array<N, AD, T> &x) const
        {
            auto res = make_array<AD, T>(radann::make_shape());
            cuda::cublas::nrm2(x.data(), res.data(), x.size());
            return res;
        }
    };
}
