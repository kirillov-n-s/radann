#pragma once
#include "../core/eager.h"
#include "../core/reduce.h"
#include "unary.h"

namespace radann::func
{
    template <typename Arg>
    inline auto sum(const expr::base<Arg>&);

    template <typename Arg>
    inline auto prod(const expr::base<Arg>&);

    template <typename Arg>
    inline auto minval(const expr::base<Arg>&);

    template <typename Arg>
    inline auto maxval(const expr::base<Arg>&);

    template <typename Arg>
    inline auto norm2(const expr::base<Arg>&);

    /*template <typename Arg>
    inline auto mean(const expr::base<Arg>&);

    template <typename Arg>
    inline auto var(const expr::base<Arg>&);

    template <typename Arg>
    inline auto stddev(const expr::base<Arg>&);*/
}

namespace radann::func
{
    template <typename Arg>
    inline auto sum(const expr::base<Arg>& arg)
    {
        return expr::make_eager(oper::sum{}, arg);
    }

    template <typename Arg>
    inline auto prod(const expr::base<Arg>& arg)
    {
        return expr::make_eager(oper::prod{}, arg);
    }

    template <typename Arg>
    inline auto minval(const expr::base<Arg>& arg)
    {
        return expr::make_eager(oper::minval{}, arg);
    }

    template <typename Arg>
    inline auto maxval(const expr::base<Arg>& arg)
    {
        return expr::make_eager(oper::maxval{}, arg);
    }

    template <typename Arg>
    inline auto norm2(const expr::base<Arg>& arg)
    {
        return expr::make_eager(oper::norm2{}, arg);
    }

    /*template <typename Arg>
    inline auto mean(const expr::base<Arg>& arg)
    {
        auto map = expr::eager(oper::sum{}, arg);
        return map /= constant<typename Arg::value_type>(map.arg().size());
    }

    template <typename Arg>
    inline auto var(const expr::base<Arg>& arg)
    {
        return sum(pow2(arg - mean(arg)));
    }

    template <typename Arg>
    inline auto stddev(const expr::base<Arg>& arg)
    {
        return make_array(sqrt(var(arg)));
    }*/
}
