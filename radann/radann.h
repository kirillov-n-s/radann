#pragma once
#include "func/unary.h"
#include "func/binary.h"
#include "func/linalg.h"
#include "func/reduce.h"
#include "func/sequence.h"
#include "func/random.h"
#include "diff/strategy_dynamic_ad.h"

namespace radann
{
    using core::shape;
    using core::make_shape;
    using core::copy;
    using core::eval;
    using core::real;
    using core::autodiff;

    template<typename T = real>
    using array = core::array<T, diff::strategy_dynamic_ad<T>>;

    using namespace func;

    template<typename T = real>
    inline auto make_array(const core::shape&, bool = autodiff);

    template<typename InputIterator>
    inline auto make_array(const core::shape&, InputIterator, InputIterator, bool = autodiff);
    template<typename T>
    inline auto make_array(const core::shape&, const std::initializer_list<T>&, bool = autodiff);

    template <typename Expr>
    inline auto make_array(const core::shape&, const expr::base<Expr>&);
    template <typename Expr>
    inline auto make_array(const expr::base<Expr>&);

    template <typename Expr>
    inline auto make_array(const core::shape&, const expr::base<Expr>&, bool);
    template <typename Expr>
    inline auto make_array(const expr::base<Expr>&, bool);

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

    template<typename T = real>
    void reverse();

    template<typename T = real>
    void clear();
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

    template<typename T>
    void reverse()
    {
        radann::diff::get_tape<T>()->reverse();
    }

    template<typename T>
    void clear()
    {
        radann::diff::get_tape<T>()->clear();
    }
}
