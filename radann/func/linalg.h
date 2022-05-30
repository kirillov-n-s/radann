#pragma once
#include "../cuda/cublas.h"
#include "../core/array.h"

namespace radann::func
{
    struct dot
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
    };

    template<bool LTrans, bool RTrans>
    struct matmul
    {
        static constexpr bool requires_validation = true;

        template<typename Lhs, typename Rhs>
        void validate(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs) const
        {
            static_assert(!RTrans || Rhs::rank == 2, "Non-matrix transposition attempt in matrix multiplication.");
            if (lhs.self().shape(!LTrans) != rhs.self().shape(RTrans))
                throw std::invalid_argument("Shape mismatch in matrix multiplication.");
        }

        template <bool ADLhs, bool ADRhs, typename T>
        auto operator()(const array<2, ADLhs, T> &x, const array<1, ADRhs, T> &y) const
        {
            auto res = make_array<ADLhs || ADRhs, T>(make_shape(x.shape(LTrans)));
            cuda::cublas::gemv<LTrans>(x.data(), y.data(), res.data(), x.shape(0), x.shape(1));
            return res;
        }

        template <bool ADLhs, bool ADRhs, typename T>
        auto operator()(const array<2, ADLhs, T> &x, const array<2, ADRhs, T> &y) const
        {
            auto rows = x.shape(LTrans);
            auto cols = y.shape(!RTrans);
            auto res = make_array<ADLhs || ADRhs, T>(make_shape(rows, cols));
            cuda::cublas::gemm<LTrans, RTrans>(x.data(), y.data(), res.data(),
                                               rows, x.shape(!LTrans), cols);
            return res;
        }

        template<typename Lhs, typename Rhs, typename Mult>
        auto accumulate_grad_lhs(const expr::base<Lhs> &lhs,
                                 const expr::base<Rhs> &rhs,
                                 const expr::base<Mult> &mult) const
        {
            return mult.self();
        }

        template<typename Lhs, typename Rhs, typename Mult>
        auto accumulate_grad_rhs(const expr::base<Lhs> &lhs,
                                 const expr::base<Rhs> &rhs,
                                 const expr::base<Mult> &mult) const
        {
            return mult.self();
        }
    };

    struct trans
    {
        static constexpr bool requires_validation = false;

        template <bool AD, typename T>
        auto operator()(const array<2, AD, T> &x) const
        {
            auto rows = x.shape(0);
            auto cols = x.shape(1);
            auto res = make_array<AD, T>(radann::make_shape(cols, rows));
            cuda::cublas::geam(x.data(), res.data(), rows, cols);
            return res;
        }
    };
}