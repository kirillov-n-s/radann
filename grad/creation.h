#pragma once
#include "array.h"
#include "sequence.h"
#include "random.h"

namespace grad
{
    template <typename Expr, typename... Extents>
    inline auto make_array(const engine::expr<Expr>&, Extents...);
    template <typename Expr>
    inline auto make_array(const engine::expr<Expr>&);

    template <typename T, typename... Extents>
    inline auto make_constant_array(T, Extents...);
    template <typename T, typename... Extents>
    inline auto make_array_of_zeroes(Extents...);
    template <typename T, typename... Extents>
    inline auto make_array_of_ones(Extents...);
    template <typename T, typename... Extents>
    inline auto make_arithm_array(T, T, Extents...);
    template <typename T, typename... Extents>
    inline auto make_geom_array(T, T, Extents...);

    template <typename T, typename... Extents>
    inline auto make_uniform_array(Extents...);
    template <typename T, typename... Extents>
    inline auto make_normal_array(Extents...);

    template <typename T, size_t N>
    inline auto make_copy(const array<T, N>&);
}

namespace grad
{
    template <typename Expr, typename... Extents>
    inline auto make_array(const engine::expr<Expr>& expr, Extents... extents)
    {
        auto shape = make_shape(extents...);
        return array<typename Expr::value_type, shape.rank> { shape, expr };
    }

    template <typename Expr>
    inline auto make_array(const engine::expr<Expr>& expr)
    {
        auto shape = expr.self().shape();
        return array<typename Expr::value_type, shape.rank> { shape, expr };
    }

    template <typename T, typename... Extents>
    inline auto make_constant_array(T value, Extents... extents)
    {
        auto shape = make_shape(extents...);
        return array<T, shape.rank> { shape, constant(value) };
    }

    template <typename T, typename... Extents>
    inline auto make_array_of_zeroes(Extents... extents)
    {
        auto shape = make_shape(extents...);
        return array<T, shape.rank> { shape };
    }

    template <typename T, typename... Extents>
    inline auto make_array_of_ones(Extents... extents)
    {
        auto shape = make_shape(extents...);
        return array<T, shape.rank> { shape, constant<T>(1) };
    }

    template <typename T, typename... Extents>
    inline auto make_arithm_array(T offset, T step, Extents... extents)
    {
        auto shape = make_shape(extents...);
        return array<T, shape.rank> { shape, arithm(offset, step) };
    }

    template <typename T, typename... Extents>
    inline auto make_geom_array(T scale, T ratio, Extents... extents)
    {
        auto shape = make_shape(extents...);
        return array<T, shape.rank> { shape, geom(scale, ratio) };
    }

    template <typename T, typename... Extents>
    inline auto make_uniform_array(Extents... extents)
    {
        auto shape = make_shape(extents...);
        return array<T, shape.rank> { shape, uniform<T>() };
    }

    template <typename T, typename... Extents>
    inline auto make_normal_array(Extents... extents)
    {
        auto shape = make_shape(extents...);
        return array<T, shape.rank> { shape, normal<T>() };
    }

    template <typename T, size_t N>
    inline auto make_copy(const array<T, N>& other)
    {
        return array<T, N> { other.shape(), other };
    }
}
