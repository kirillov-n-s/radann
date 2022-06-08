#pragma once
#include "../func/unary.h"
#include "../func/sequence.h"

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
}
