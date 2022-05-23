#pragma once
#include "../core/unary.h"

namespace radann::functor
{
    struct add
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x, T y) const
        {
            return x + y;
        }

        template<typename Lhs, typename Rhs, typename Mult>
        auto accumulate_grad_lhs(const engine::expr<Lhs> &lhs,
                                 const engine::expr<Rhs> &rhs,
                                 const engine::expr<Mult> &mult) const
        {
            return mult.self();
        }

        template<typename Lhs, typename Rhs, typename Mult>
        auto accumulate_grad_rhs(const engine::expr<Lhs> &lhs,
                                 const engine::expr<Rhs> &rhs,
                                 const engine::expr<Mult> &mult) const
        {
            return mult.self();
        }
    };

    struct sub
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x, T y) const
        {
            return x - y;
        }

        template<typename Lhs, typename Rhs, typename Mult>
        auto accumulate_grad_lhs(const engine::expr<Lhs> &lhs,
                                 const engine::expr<Rhs> &rhs,
                                 const engine::expr<Mult> &mult) const
        {
            return mult.self();
        }

        template<typename Lhs, typename Rhs, typename Mult>
        auto accumulate_grad_rhs(const engine::expr<Lhs> &lhs,
                                 const engine::expr<Rhs> &rhs,
                                 const engine::expr<Mult> &mult) const
        {
            return -mult;
        }
    };

    struct mul
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x, T y) const
        {
            return x * y;
        }

        template<typename Lhs, typename Rhs, typename Mult>
        auto accumulate_grad_lhs(const engine::expr<Lhs> &lhs,
                                 const engine::expr<Rhs> &rhs,
                                 const engine::expr<Mult> &mult) const
        {
            return rhs * mult;
        }

        template<typename Lhs, typename Rhs, typename Mult>
        auto accumulate_grad_rhs(const engine::expr<Lhs> &lhs,
                                 const engine::expr<Rhs> &rhs,
                                 const engine::expr<Mult> &mult) const
        {
            return lhs * mult;
        }
    };

    struct div
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x, T y) const
        {
            return x / y;
        }

        template<typename Lhs, typename Rhs, typename Mult>
        auto accumulate_grad_lhs(const engine::expr<Lhs> &lhs,
                                 const engine::expr<Rhs> &rhs,
                                 const engine::expr<Mult> &mult) const
        {
            return mult / rhs;
        }

        template<typename Lhs, typename Rhs, typename Mult>
        auto accumulate_grad_rhs(const engine::expr<Lhs> &lhs,
                                 const engine::expr<Rhs> &rhs,
                                 const engine::expr<Mult> &mult) const
        {
            return -lhs / radann::pow2(rhs) * mult;
        }
    };

    struct pow
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x, T y) const
        {
            return ::pow(x, y);
        }
    };

    struct atan2
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x, T y) const
        {
            return ::atan2(x, y);
        }
    };
    
    struct min
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x, T y) const
        {
            return ::fmin(x, y);
        }
    };
    
    struct max
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x, T y) const
        {
            return ::fmax(x, y);
        }
    };
}
