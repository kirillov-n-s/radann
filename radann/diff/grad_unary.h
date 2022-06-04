#pragma once
#include "../meta/meta.h"
#include "../func/unary.h"
#include "../func/sequence.h"

namespace radann::diff
{
    template<typename Op>
    struct grad
    {
        static_assert(meta::always_false_v<Op>, "Operator type is not supported.");
    };

    template<>
    struct grad<radann::oper::neg>
    {
        template<typename Arg, typename Mult>
        auto operator()(const expr::base<Arg> &arg, const expr::base<Mult> &mult) const
        {
            return -mult;
        }
    };

    template<>
    struct grad<radann::oper::sin>
    {
        template<typename Arg, typename Mult>
        auto operator()(const expr::base<Arg> &arg, const expr::base<Mult> &mult) const
        {
            return func::cos(arg) * mult;
        }
    };

    template<>
    struct grad<radann::oper::pow2>
    {
        template<typename Arg, typename Mult>
        auto operator()(const expr::base<Arg> &arg, const expr::base<Mult> &mult) const
        {
            return func::constant<typename Arg::value_type>(2) * arg * mult;
        }
    };

    template<>
    struct grad<radann::oper::log>
    {
        template<typename Arg, typename Mult>
        auto operator()(const expr::base<Arg> &arg, const expr::base<Mult> &mult) const
        {
            return mult / arg;
        }
    };
}
