#pragma once
#include "../cuda/reduce.h"
//#include "../cuda/cublas.h"
#include "array.h"
#include "../oper/binary.h"

namespace radann::core
{
    struct sum
    {
        static constexpr bool does_validate = false;

        template <typename T, typename Strategy>
        auto operator()(const array<T, Strategy> &x) const
        {
            auto res = array<T, Strategy> { make_shape() };
            cuda::reduce(x.data(), res.data(), x.size(), oper::add{});
            return res;
        }
    };

    /*struct maxval
    {
        static constexpr bool does_validate = false;

        template <typename T, typename Strategy>
        auto operator()(const array<T, Strategy> &x) const
        {
            auto res = array<T, Strategy> { make_shape() };
            cuda::reduce(x.data(), res.data(), x.size(), oper::max{});
            return res;
        }
    };

    struct minval
    {
        static constexpr bool does_validate = false;

        template <typename T, typename Strategy>
        auto operator()(const array<T, Strategy> &x) const
        {
            auto res = array<T, Strategy> { make_shape() };
            cuda::reduce(x.data(), res.data(), x.size(), oper::min{});
            return res;
        }
    };*/

    /*struct prod
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
    };*/
}
