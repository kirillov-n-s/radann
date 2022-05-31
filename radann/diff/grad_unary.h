#pragma once
#include "../core/util.h"
#include "../core/unary.h"
#include "../core/sequence.h"

namespace radann::diff
{
    template<typename Op>
    struct grad
    {
        static_assert(radann::always_false_v<Op>, "Operator type is not supported.");
    };

    template<>
    struct grad<radann::func::neg>
    {
        template<typename Arg, typename Mult>
        auto operator()(const expr::base<Arg> &arg, const expr::base<Mult> &mult) const
        {
            return -mult;
        }
    };

    template<>
    struct grad<radann::func::sin>
    {
        template<typename Arg, typename Mult>
        auto operator()(const expr::base<Arg> &arg, const expr::base<Mult> &mult) const
        {
            return radann::cos(arg) * mult;
        }
    };

    template<>
    struct grad<radann::func::pow2>
    {
        template<typename Arg, typename Mult>
        auto operator()(const expr::base<Arg> &arg, const expr::base<Mult> &mult) const
        {
            return radann::constant<typename Arg::value_type>(2) * arg * mult;
        }
    };

    template<>
    struct grad<radann::func::log>
    {
        template<typename Arg, typename Mult>
        auto operator()(const expr::base<Arg> &arg, const expr::base<Mult> &mult) const
        {
            return mult / arg;
        }
    };
}
