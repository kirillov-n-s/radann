#pragma once
#include "../cuda/cublas.h"
#include "array.h"

namespace radann::core
{
    /*struct dot
    {
        static constexpr bool requires_validation = true;

        template<typename Lhs, typename Rhs>
        void validate(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs) const
        {
            if (lhs.self().shape() != rhs.self().shape())
                throw std::invalid_argument("Shape mismatch in dot product.");
        }

        template <size_t N, bool ADLhs, bool ADRhs, typename T>
        auto operator()(const array<N, ADLhs, T> &x, const array<N, ADRhs, T> &y) const
        {
            auto res = make_array<ADLhs || ADRhs, T>(radann::make_shape());
            cuda::cublas::dot(x.data(), y.data(), res.data(), x.size());
            return res;
        }
    };

    struct outer
    {
        static constexpr bool requires_validation = false;

        template <bool ADLhs, bool ADRhs, typename T>
        auto operator()(const array<1, ADLhs, T> &x, const array<1, ADRhs, T> &y) const
        {
            auto rows = x.size();
            auto cols = y.size();
            auto res = make_array<ADLhs || ADRhs, T>(radann::make_shape(rows, cols));
            cuda::cublas::ger(x.data(), y.data(), res.data(), rows, cols);
            return res;
        }
    };*/

    template<bool LTrans, bool RTrans>
    struct matmul
    {
        struct backward_lhs {};
        struct backward_rhs {};

        static constexpr bool does_validate = true;

        template<typename Lhs, typename Rhs>
        void validate(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs) const
        {
            auto lself = lhs.self();
            auto rself = rhs.self();

            auto lrank = lself.rank();
            auto rrank = rself.rank();

            auto lcols = lself.shape(!LTrans);
            auto rrows = rself.shape(RTrans);

            if (lrank > 2 || rrank > lrank || rrank < 1 || lcols != rrows)
                throw std::invalid_argument("Illegal shape in matrix multiplication.");
        }

        template <typename T, typename Strategy>
        auto operator()(const array<T, Strategy> &x, const array<T, Strategy> &y) const
        {
            auto xrank = x.rank();
            auto yrank = y.rank();

            if (xrank > yrank)
            {
                auto res = array<T, Strategy> { make_shape(x.shape(LTrans)) };
                cuda::cublas::gemv<LTrans>(x.data(), y.data(), res.data(), x.shape(0), x.shape(1));
                return res;
            }

            if (xrank == 1)
            {
                auto rows = x.size();
                auto cols = y.size();
                auto res = array<T, Strategy> { make_shape(rows, cols) };
                cuda::cublas::ger(x.data(), y.data(), res.data(), rows, cols);
                return res;
            }

            auto rows = x.shape(LTrans);
            auto mid = x.shape(!LTrans);
            auto cols = y.shape(!RTrans);
            auto res = array<T, Strategy> { make_shape(rows, cols) };
            cuda::cublas::gemm<LTrans, RTrans>(x.data(), y.data(), res.data(), rows, mid, cols);
            return res;
        }
    };

    /*struct trans
    {
        static constexpr bool requires_validation = false;

        template <typename T>
        auto operator()(const array<T> &x) const
        {
            auto rows = x.shape(0);
            auto cols = x.shape(1);
            auto res = make_array<T>(core::make_shape(cols, rows));
            cuda::cublas::geam(x.data(), res.data(), rows, cols);
            return res;
        }
    };*/
}
