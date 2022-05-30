#pragma once
#include "../expr/unary.h"
#include "../func/unary.h"

namespace radann
{
    template <typename Arg>
    inline auto operator-(const expr::base<Arg>&);

    template <typename Arg>
    inline auto abs(const expr::base<Arg>&);

    template <typename Arg>
    inline auto sqrt(const expr::base<Arg>&);

    template <typename Arg>
    inline auto cbrt(const expr::base<Arg>&);

    template <typename Arg>
    inline auto sin(const expr::base<Arg>&);

    template <typename Arg>
    inline auto cos(const expr::base<Arg>&);

    template <typename Arg>
    inline auto tan(const expr::base<Arg>&);

    template <typename Arg>
    inline auto asin(const expr::base<Arg>&);

    template <typename Arg>
    inline auto acos(const expr::base<Arg>&);

    template <typename Arg>
    inline auto atan(const expr::base<Arg>&);

    template <typename Arg>
    inline auto sinh(const expr::base<Arg>&);

    template <typename Arg>
    inline auto cosh(const expr::base<Arg>&);

    template <typename Arg>
    inline auto tanh(const expr::base<Arg>&);

    template <typename Arg>
    inline auto asinh(const expr::base<Arg>&);

    template <typename Arg>
    inline auto acosh(const expr::base<Arg>&);

    template <typename Arg>
    inline auto atanh(const expr::base<Arg>&);

    template <typename Arg>
    inline auto exp(const expr::base<Arg>&);

    template <typename Arg>
    inline auto exp2(const expr::base<Arg>&);

    template <typename Arg>
    inline auto exp10(const expr::base<Arg>&);

    template <typename Arg>
    inline auto expm1(const expr::base<Arg>&);

    template <typename Arg>
    inline auto log(const expr::base<Arg>&);

    template <typename Arg>
    inline auto log2(const expr::base<Arg>&);

    template <typename Arg>
    inline auto log10(const expr::base<Arg>&);

    template <typename Arg>
    inline auto log1p(const expr::base<Arg>&);

    template <typename Arg>
    inline auto sigmoid(const expr::base<Arg>&);

    template <typename Arg>
    inline auto sign(const expr::base<Arg>&);

    template <typename Arg>
    inline auto pow2(const expr::base<Arg>&);
}

namespace radann
{
    template <typename Arg>
    inline auto operator-(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::neg{}, arg);
    }

    template <typename Arg>
    inline auto abs(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::abs{}, arg);
    }

    template <typename Arg>
    inline auto sqrt(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::sqrt{}, arg);
    }

    template <typename Arg>
    inline auto cbrt(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::cbrt{}, arg);
    }

    template <typename Arg>
    inline auto sin(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::sin{}, arg);
    }

    template <typename Arg>
    inline auto cos(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::cos{}, arg);
    }

    template <typename Arg>
    inline auto tan(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::tan{}, arg);
    }

    template <typename Arg>
    inline auto asin(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::asin{}, arg);
    }

    template <typename Arg>
    inline auto acos(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::acos{}, arg);
    }

    template <typename Arg>
    inline auto atan(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::atan{}, arg);
    }

    template <typename Arg>
    inline auto sinh(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::sinh{}, arg);
    }

    template <typename Arg>
    inline auto cosh(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::cosh{}, arg);
    }

    template <typename Arg>
    inline auto tanh(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::tanh{}, arg);
    }

    template <typename Arg>
    inline auto asinh(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::asinh{}, arg);
    }

    template <typename Arg>
    inline auto acosh(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::acosh{}, arg);
    }

    template <typename Arg>
    inline auto atanh(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::atanh{}, arg);
    }

    template <typename Arg>
    inline auto exp(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::exp{}, arg);
    }

    template <typename Arg>
    inline auto exp2(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::exp2{}, arg);
    }

    template <typename Arg>
    inline auto exp10(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::exp10{}, arg);
    }

    template <typename Arg>
    inline auto expm1(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::expm1{}, arg);
    }

    template <typename Arg>
    inline auto log(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::log{}, arg);
    }

    template <typename Arg>
    inline auto log2(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::log2{}, arg);
    }

    template <typename Arg>
    inline auto log10(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::log10{}, arg);
    }

    template <typename Arg>
    inline auto log1p(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::log1p{}, arg);
    }

    template <typename Arg>
    inline auto sigmoid(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::sigmoid{}, arg);
    }

    template <typename Arg>
    inline auto sign(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::sgn{}, arg);
    }

    template <typename Arg>
    inline auto pow2(const expr::base<Arg> &arg)
    {
        return expr::make_lazy(func::pow2{}, arg);
    }
}
