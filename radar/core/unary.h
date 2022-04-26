#pragma once
#include "engine/unary_lazy.h"
#include "functor/unary.h"

namespace radar
{
    template <typename Arg>
    inline auto operator-(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto abs(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto sqrt(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto cbrt(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto sin(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto cos(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto tan(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto asin(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto acos(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto atan(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto sinh(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto cosh(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto tanh(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto asinh(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto acosh(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto atanh(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto exp(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto exp2(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto exp10(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto expm1(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto log(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto log2(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto log10(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto log1p(const engine::expr<Arg>&);
}

namespace radar
{
    template <typename Arg>
    inline auto operator-(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::neg{}, arg);
    }

    template <typename Arg>
    inline auto abs(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::abs{}, arg);
    }

    template <typename Arg>
    inline auto sqrt(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::sqrt{}, arg);
    }

    template <typename Arg>
    inline auto cbrt(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::cbrt{}, arg);
    }

    template <typename Arg>
    inline auto sin(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::sin{}, arg);
    }

    template <typename Arg>
    inline auto cos(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::cos{}, arg);
    }

    template <typename Arg>
    inline auto tan(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::tan{}, arg);
    }

    template <typename Arg>
    inline auto asin(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::asin{}, arg);
    }

    template <typename Arg>
    inline auto acos(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::acos{}, arg);
    }

    template <typename Arg>
    inline auto atan(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::atan{}, arg);
    }

    template <typename Arg>
    inline auto sinh(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::sinh{}, arg);
    }

    template <typename Arg>
    inline auto cosh(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::cosh{}, arg);
    }

    template <typename Arg>
    inline auto tanh(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::tanh{}, arg);
    }

    template <typename Arg>
    inline auto asinh(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::asinh{}, arg);
    }

    template <typename Arg>
    inline auto acosh(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::acosh{}, arg);
    }

    template <typename Arg>
    inline auto atanh(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::atanh{}, arg);
    }

    template <typename Arg>
    inline auto exp(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::exp{}, arg);
    }

    template <typename Arg>
    inline auto exp2(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::exp2{}, arg);
    }

    template <typename Arg>
    inline auto exp10(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::exp10{}, arg);
    }

    template <typename Arg>
    inline auto expm1(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::expm1{}, arg);
    }

    template <typename Arg>
    inline auto log(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::log{}, arg);
    }

    template <typename Arg>
    inline auto log2(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::log2{}, arg);
    }

    template <typename Arg>
    inline auto log10(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::log10{}, arg);
    }

    template <typename Arg>
    inline auto log1p(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::log1p{}, arg);
    }
}
