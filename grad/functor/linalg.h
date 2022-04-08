#pragma once
#include "../cuda/cublas.h"
#include "../creation.h"

namespace grad::functor
{
    struct dot
    {
        static constexpr bool requires_validation = true;

        template<typename Lhs, typename Rhs>
        void validate(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs) const
        {
            if (lhs.self().shape() != rhs.self().shape())
                throw std::invalid_argument("Shape mismatch in dot product.");
        }

        template <typename T, size_t N>
        inline auto operator()(const array<T, N> &x, const array<T, N> &y) const
        {
            auto res = make_zeroes<T>(grad::make_shape());
            cuda::cublas::dot(x.data(), y.data(),
                              res.data(),
                              x.size());
            return res;
        }
    };

    struct outer
    {
        static constexpr bool requires_validation = false;

        template <typename T>
        inline auto operator()(const array<T, 1> &x, const array<T, 1> &y) const
        {
            auto rows = x.size();
            auto cols = y.size();
            auto res = make_zeroes<T>(grad::make_shape(rows, cols));
            cuda::cublas::ger(x.data(), y.data(),
                              res.data(),
                              rows, cols);
            return res;
        }
    };

    struct matmul
    {
        static constexpr bool requires_validation = true;

        template<typename Lhs, typename Rhs>
        void validate(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs) const
        {
            if (lhs.self().shape(1) != rhs.self().shape(0))
                throw std::invalid_argument("Shape mismatch in matrix multiplication.");
        }

        template <typename T>
        inline auto operator()(const array<T, 2> &x, const array<T, 1> &y) const
        {
            auto rows = x.shape(0);
            auto res = make_zeroes<T>(make_shape(rows));
            cuda::cublas::gemv(x.data(), y.data(),
                               res.data(),
                               rows, x.shape(1));
            return res;
        }

        template <typename T>
        inline auto operator()(const array<T, 2> &x, const array<T, 2> &y) const
        {
            auto rows = x.shape(0);
            auto cols = y.shape(1);
            auto res = make_zeroes<T>(make_shape(rows, cols));
            cuda::cublas::gemm(x.data(), y.data(),
                               res.data(),
                               rows, x.shape(1), cols);
            return res;
        }
    };

    struct trans
    {
        static constexpr bool requires_validation = false;

        template <typename T>
        inline auto operator()(const array<T, 2> &x) const
        {
            auto rows = x.shape(0);
            auto cols = x.shape(1);
            auto res = make_zeroes<T>(grad::make_shape(cols, rows));
            cuda::cublas::geam(x.data(),
                               res.data(),
                               rows, cols);
            return res;
        }
    };
}
