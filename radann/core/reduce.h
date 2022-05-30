#pragma once
#include "../expr/eager.h"
#include "../func/reduce.h"
#include "unary.h"

namespace radann
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

namespace radann
{
    template <typename Arg>
    inline auto sum(const expr::base<Arg>& arg)
    {
        return expr::make_eager(func::sum{}, arg);
    }

    template <typename Arg>
    inline auto prod(const expr::base<Arg>& arg)
    {
        return expr::make_eager(func::prod{}, arg);
    }

    template <typename Arg>
    inline auto minval(const expr::base<Arg>& arg)
    {
        return expr::make_eager(func::minval{}, arg);
    }

    template <typename Arg>
    inline auto maxval(const expr::base<Arg>& arg)
    {
        return expr::make_eager(func::maxval{}, arg);
    }

    template <typename Arg>
    inline auto norm2(const expr::base<Arg>& arg)
    {
        return expr::make_eager(func::norm2{}, arg);
    }

    /*template <typename Arg>
    inline auto mean(const expr::base<Arg>& arg)
    {
        auto map = expr::make_eager(func::sum{}, arg);
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
