#pragma once
#include "../func/sequence.h"
#include "../func/random.h"
#include "using.h"

namespace radann
{
    template<typename T = real>
    inline auto make_array(const shape&, bool = autodiff);

    template<typename InputIterator>
    inline auto make_array(const shape&, InputIterator, InputIterator, bool = autodiff);
    template<typename T>
    inline auto make_array(const shape&, const std::initializer_list<T>&, bool = autodiff);

    template <typename Expr>
    inline auto make_array(const shape&, const expr::base<Expr>&);
    template <typename Expr>
    inline auto make_array(const expr::base<Expr>&);

    template <typename Expr>
    inline auto make_array(const shape&, const expr::base<Expr>&, bool);
    template <typename Expr>
    inline auto make_array(const expr::base<Expr>&, bool);

    template <typename T>
    inline auto copy(const array<T>&);
    template <typename T>
    inline auto copy(const array<T>&, bool ad);

    template <typename T>
    inline auto make_constant(const shape&, T, bool = autodiff);
    template <typename T = real>
    inline auto make_ones(const shape&, bool = autodiff);
    template <typename T>
    inline auto make_arithm(const shape&, T, T, bool = autodiff);
    template <typename T>
    inline auto make_geom(const shape&, T, T, bool = autodiff);

    template <typename T = real>
    inline auto make_uniform(const shape&, bool = autodiff, unsigned int = std::random_device{}());
    template <typename T = real>
    inline auto make_normal(const shape&, bool = autodiff, unsigned int = std::random_device{}());
}

namespace radann
{
    template<typename T>
    inline auto make_array(const shape& shape, bool ad)
    {
        return array<T> { shape, ad };
    }

    template<typename InputIterator>
    inline auto make_array(const shape& shape, InputIterator first, InputIterator last, bool ad)
    {
        return array<typename std::iterator_traits<InputIterator>::value_type> { shape, first, last, ad };
    }

    template<typename T>
    inline auto make_array(const shape& shape, const std::initializer_list<T>& data, bool ad)
    {
        return array<T> { shape, data, ad };
    }

    template<typename Expr>
    inline auto make_array(const shape& shape, const expr::base<Expr>& expr)
    {
        return array<typename Expr::value_type> { shape, expr };
    }

    template<typename Expr>
    inline auto make_array(const expr::base<Expr>& expr)
    {
        return array<typename Expr::value_type> { expr };
    }

    template<typename Expr>
    inline auto make_array(const shape& shape, const expr::base<Expr>& expr, bool ad)
    {
        return array<typename Expr::value_type> { shape, expr, ad };
    }

    template<typename Expr>
    inline auto make_array(const expr::base<Expr>& expr, bool ad)
    {
        return array<typename Expr::value_type> { expr, ad };
    }

    template <typename T>
    inline auto copy(const array<T> &other)
    {
        return array<T> { other.data(), other.shape(), other.ad() };
    }

    template <typename T>
    inline auto copy(const array<T> &other, bool ad)
    {
        return array<T> { other.data(), other.shape(), ad };
    }

    template <typename T>
    inline auto make_constant(const shape& shape, T value, bool ad)
    {
        return array<T> { shape, constant(value), ad };
    }

    template <typename T>
    inline auto make_ones(const shape& shape, bool ad)
    {
        return array<T> { shape, constant<T>(1), ad };
    }

    template <typename T>
    inline auto make_arithm(const shape& shape, T offset, T step, bool ad)
    {
        return array<T> { shape, arithm(offset, step), ad };
    }

    template <typename T>
    inline auto make_geom(const shape& shape, T scale, T ratio, bool ad)
    {
        return array<T> { shape, geom(scale, ratio), ad };
    }

    template <typename T>
    inline auto make_uniform(const shape& shape, bool ad, unsigned int seed)
    {
        return array<T> { shape, uniform<T>(seed), ad };
    }

    template <typename T>
    inline auto make_normal(const shape& shape, bool ad, unsigned int seed)
    {
        return array<T> { shape, normal<T>(seed), ad };
    }
}
