#pragma once
#include "../func/unary.h"
#include "../func/sequence.h"
#include "../func/reduce.h"

namespace radann::diff
{
    template<typename Arg, typename Mult>
    auto grad(const expr::base<Arg> &arg, const expr::base<Mult> &mult, const oper::neg&)
    {
        return -mult;
    }

    template<typename Arg, typename Mult>
    auto grad(const expr::base<Arg> &arg, const expr::base<Mult> &mult, const oper::sin&)
    {
        return func::cos(arg) * mult;
    }

    template<typename Arg, typename Mult>
    auto grad(const expr::base<Arg> &arg, const expr::base<Mult> &mult, const oper::pow2&)
    {
        return func::constant<typename Arg::value_type>(2) * arg * mult;
    }

    template<typename Arg, typename Mult>
    auto grad(const expr::base<Arg> &arg, const expr::base<Mult> &mult, const oper::log&)
    {
        return mult / arg;
    }

    template<typename Arg, typename Mult>
    auto grad(const expr::base<Arg> &arg, const expr::base<Mult> &mult, const oper::exp&)
    {
        return func::exp(arg) * mult;
    }

    template<typename Arg, typename Mult>
    auto grad(const expr::base<Arg> &arg, const expr::base<Mult> &mult, const oper::sigmoid&)
    {
        auto tmp = func::sigmoid(arg);
        return tmp * (func::constant<typename Arg::value_type>(1) - tmp) * mult;
    }

    template<typename Arg, typename Mult>
    auto grad(const expr::base<Arg> &arg, const expr::base<Mult> &mult, const oper::tanh&)
    {
        auto tmp = func::tanh(arg);
        return (func::constant<typename Arg::value_type>(1) - func::pow2(tmp)) * mult;
    }

    template<typename Arg, typename Mult>
    auto grad(const expr::base<Arg> &arg, const expr::base<Mult> &mult, const core::sum&)
    {
        return mult.self();
    }
}
