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
        return expr::make_expr(func::neg{}, arg);
    }

    template <typename Arg>
    inline auto abs(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::abs{}, arg);
    }

    template <typename Arg>
    inline auto sqrt(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::sqrt{}, arg);
    }

    template <typename Arg>
    inline auto cbrt(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::cbrt{}, arg);
    }

    template <typename Arg>
    inline auto sin(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::sin{}, arg);
    }

    template <typename Arg>
    inline auto cos(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::cos{}, arg);
    }

    template <typename Arg>
    inline auto tan(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::tan{}, arg);
    }

    template <typename Arg>
    inline auto asin(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::asin{}, arg);
    }

    template <typename Arg>
    inline auto acos(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::acos{}, arg);
    }

    template <typename Arg>
    inline auto atan(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::atan{}, arg);
    }

    template <typename Arg>
    inline auto sinh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::sinh{}, arg);
    }

    template <typename Arg>
    inline auto cosh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::cosh{}, arg);
    }

    template <typename Arg>
    inline auto tanh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::tanh{}, arg);
    }

    template <typename Arg>
    inline auto asinh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::asinh{}, arg);
    }

    template <typename Arg>
    inline auto acosh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::acosh{}, arg);
    }

    template <typename Arg>
    inline auto atanh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::atanh{}, arg);
    }

    template <typename Arg>
    inline auto exp(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::exp{}, arg);
    }

    template <typename Arg>
    inline auto exp2(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::exp2{}, arg);
    }

    template <typename Arg>
    inline auto exp10(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::exp10{}, arg);
    }

    template <typename Arg>
    inline auto expm1(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::expm1{}, arg);
    }

    template <typename Arg>
    inline auto log(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::log{}, arg);
    }

    template <typename Arg>
    inline auto log2(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::log2{}, arg);
    }

    template <typename Arg>
    inline auto log10(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::log10{}, arg);
    }

    template <typename Arg>
    inline auto log1p(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::log1p{}, arg);
    }

    template <typename Arg>
    inline auto sigmoid(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::sigmoid{}, arg);
    }

    template <typename Arg>
    inline auto sign(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::sgn{}, arg);
    }

    template <typename Arg>
    inline auto pow2(const expr::base<Arg> &arg)
    {
        return expr::make_expr(func::pow2{}, arg);
    }
}
