#pragma once
#include "ops/unary.h"
#include "ops/unary_functors.h"

namespace grad
{
    template <typename Arg>
    inline auto operator-(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto abs(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto sqrt(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto cbrt(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto sin(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto cos(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto tan(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto asin(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto acos(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto atan(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto sinh(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto cosh(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto tanh(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto asinh(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto acosh(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto atanh(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto exp(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto exp2(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto exp10(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto expm1(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto log(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto log2(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto log10(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto log1p(const ops::expr<Arg>&);
}

namespace grad
{
    template <typename Arg>
    inline auto operator-(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::neg{}, arg);
    }

    template <typename Arg>
    inline auto abs(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::abs{}, arg);
    }

    template <typename Arg>
    inline auto sqrt(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::sqrt{}, arg);
    }

    template <typename Arg>
    inline auto cbrt(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::cbrt{}, arg);
    }

    template <typename Arg>
    inline auto sin(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::sin{}, arg);
    }

    template <typename Arg>
    inline auto cos(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::cos{}, arg);
    }

    template <typename Arg>
    inline auto tan(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::tan{}, arg);
    }

    template <typename Arg>
    inline auto asin(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::asin{}, arg);
    }

    template <typename Arg>
    inline auto acos(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::acos{}, arg);
    }

    template <typename Arg>
    inline auto atan(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::atan{}, arg);
    }

    template <typename Arg>
    inline auto sinh(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::sinh{}, arg);
    }

    template <typename Arg>
    inline auto cosh(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::cosh{}, arg);
    }

    template <typename Arg>
    inline auto tanh(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::tanh{}, arg);
    }

    template <typename Arg>
    inline auto asinh(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::asinh{}, arg);
    }

    template <typename Arg>
    inline auto acosh(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::acosh{}, arg);
    }

    template <typename Arg>
    inline auto atanh(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::atanh{}, arg);
    }

    template <typename Arg>
    inline auto exp(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::exp{}, arg);
    }

    template <typename Arg>
    inline auto exp2(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::exp2{}, arg);
    }

    template <typename Arg>
    inline auto exp10(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::exp10{}, arg);
    }

    template <typename Arg>
    inline auto expm1(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::expm1{}, arg);
    }

    template <typename Arg>
    inline auto log(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::log{}, arg);
    }

    template <typename Arg>
    inline auto log2(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::log2{}, arg);
    }

    template <typename Arg>
    inline auto log10(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::log10{}, arg);
    }

    template <typename Arg>
    inline auto log1p(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::log1p{}, arg);
    }
}
