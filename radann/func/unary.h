#pragma once
#include "../expr/unary.h"
#include "../oper/unary.h"

namespace radann::func
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

namespace radann::func
{
    template <typename Arg>
    inline auto operator-(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::neg{}, arg);
    }

    template <typename Arg>
    inline auto abs(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::abs{}, arg);
    }

    template <typename Arg>
    inline auto sqrt(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::sqrt{}, arg);
    }

    template <typename Arg>
    inline auto cbrt(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::cbrt{}, arg);
    }

    template <typename Arg>
    inline auto sin(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::sin{}, arg);
    }

    template <typename Arg>
    inline auto cos(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::cos{}, arg);
    }

    template <typename Arg>
    inline auto tan(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::tan{}, arg);
    }

    template <typename Arg>
    inline auto asin(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::asin{}, arg);
    }

    template <typename Arg>
    inline auto acos(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::acos{}, arg);
    }

    template <typename Arg>
    inline auto atan(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::atan{}, arg);
    }

    template <typename Arg>
    inline auto sinh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::sinh{}, arg);
    }

    template <typename Arg>
    inline auto cosh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::cosh{}, arg);
    }

    template <typename Arg>
    inline auto tanh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::tanh{}, arg);
    }

    template <typename Arg>
    inline auto asinh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::asinh{}, arg);
    }

    template <typename Arg>
    inline auto acosh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::acosh{}, arg);
    }

    template <typename Arg>
    inline auto atanh(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::atanh{}, arg);
    }

    template <typename Arg>
    inline auto exp(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::exp{}, arg);
    }

    template <typename Arg>
    inline auto exp2(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::exp2{}, arg);
    }

    template <typename Arg>
    inline auto exp10(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::exp10{}, arg);
    }

    template <typename Arg>
    inline auto expm1(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::expm1{}, arg);
    }

    template <typename Arg>
    inline auto log(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::log{}, arg);
    }

    template <typename Arg>
    inline auto log2(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::log2{}, arg);
    }

    template <typename Arg>
    inline auto log10(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::log10{}, arg);
    }

    template <typename Arg>
    inline auto log1p(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::log1p{}, arg);
    }

    template <typename Arg>
    inline auto sigmoid(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::sigmoid{}, arg);
    }

    template <typename Arg>
    inline auto sign(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::sgn{}, arg);
    }

    template <typename Arg>
    inline auto pow2(const expr::base<Arg> &arg)
    {
        return expr::make_expr(oper::pow2{}, arg);
    }
}

using radann::func::operator-;
