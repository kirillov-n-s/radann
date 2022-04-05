#pragma once
#include "array.h"
#include "sequence.h"
#include "random.h"

namespace grad
{
    template<typename InputIterator, size_t N>
    inline auto make_array(const grad::shape<N>&, InputIterator, InputIterator);
    template<typename T, size_t N>
    inline auto make_array(const grad::shape<N>&, const std::initializer_list<T>&);

    template <typename Expr, size_t N>
    inline auto make_array(const shape<N>&, const engine::expr<Expr>&);
    template <typename Expr>
    inline auto make_array(const engine::expr<Expr>&);

    template <typename T, size_t N>
    inline auto make_constant(const shape<N>&, T);
    template <typename T, size_t N>
    inline auto make_zeroes(const shape<N>&);
    template <typename T, size_t N>
    inline auto make_ones(const shape<N>&);
    template <typename T, size_t N>
    inline auto make_arithm(const shape<N>&, T, T);
    template <typename T, size_t N>
    inline auto make_geom(const shape<N>&, T, T);

    template <typename T, size_t N>
    inline auto make_uniform(const shape<N>&);
    template <typename T, size_t N>
    inline auto make_normal(const shape<N>&);

    template <typename T, size_t N>
    inline auto make_copy(const array<T, N>&);
}

namespace grad
{
    template<typename InputIterator, size_t N>
    inline auto make_array(const grad::shape<N>& shape, InputIterator first, InputIterator last)
    {
        return array<typename std::iterator_traits<InputIterator>::value_type, N> { shape, first, last };
    }

    template<typename T, size_t N>
    inline auto make_array(const grad::shape<N>& shape, const std::initializer_list<T>& data)
    {
        return array<T, N> { shape, data };
    }

    template <typename Expr, size_t N>
    inline auto make_array(const shape<N>& shape, const engine::expr<Expr>& expr)
    {
        return array<typename Expr::value_type, shape.rank> { shape, expr };
    }

    template <typename Expr>
    inline auto make_array(const engine::expr<Expr>& expr)
    {
        auto shape = expr.self().shape();
        return array<typename Expr::value_type, shape.rank> { shape, expr };
    }

    template <typename T, size_t N>
    inline auto make_constant(const shape<N>& shape, T value)
    {
        return array<T, shape.rank> { shape, constant(value) };
    }

    template <typename T, size_t N>
    inline auto make_zeroes(const shape<N>& shape)
    {
        return array<T, shape.rank> { shape };
    }

    template <typename T, size_t N>
    inline auto make_ones(const shape<N>& shape)
    {
        return array<T, shape.rank> { shape, constant<T>(1) };
    }

    template <typename T, size_t N>
    inline auto make_arithm(const shape<N>& shape, T offset, T step)
    {
        return array<T, shape.rank> { shape, arithm(offset, step) };
    }

    template <typename T, size_t N>
    inline auto make_geom(const shape<N>& shape, T scale, T ratio)
    {
        return array<T, shape.rank> { shape, geom(scale, ratio) };
    }

    template <typename T, size_t N>
    inline auto make_uniform(const shape<N>& shape)
    {
        return array<T, shape.rank> { shape, uniform<T>() };
    }

    template <typename T, size_t N>
    inline auto make_normal(const shape<N>& shape)
    {
        return array<T, shape.rank> { shape, normal<T>() };
    }

    template <typename T, size_t N>
    inline auto make_copy(const array<T, N>& other)
    {
        return array<T, N> { other.shape(), other };
    }
}
