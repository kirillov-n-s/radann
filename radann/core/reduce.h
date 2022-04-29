#pragma once
#include "../engine/unary_eager.h"
#include "../functor/reduce.h"
#include "extension.h"

namespace radann
{
    template <typename Arg>
    inline auto sum(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto prod(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto minval(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto maxval(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto norm2(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto mean(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto var(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto stddev(const engine::expr<Arg>&);
}

namespace radann
{
    template <typename Arg>
    inline auto sum(const engine::expr<Arg>& arg)
    {
        return engine::make_eager(functor::sum{}, arg).result();
    }

    template <typename Arg>
    inline auto prod(const engine::expr<Arg>& arg)
    {
        return engine::make_eager(functor::prod{}, arg).result();
    }

    template <typename Arg>
    inline auto minval(const engine::expr<Arg>& arg)
    {
        return engine::make_eager(functor::minval{}, arg).result();
    }

    template <typename Arg>
    inline auto maxval(const engine::expr<Arg>& arg)
    {
        return engine::make_eager(functor::maxval{}, arg).result();
    }

    template <typename Arg>
    inline auto norm2(const engine::expr<Arg>& arg)
    {
        return engine::make_eager(functor::norm2{}, arg).result();
    }

    template <typename Arg>
    inline auto mean(const engine::expr<Arg>& arg)
    {
        auto map = engine::make_eager(functor::sum{}, arg);
        return map.result() /= constant<typename Arg::value_type>(map.arg().size());
    }

    template <typename Arg>
    inline auto var(const engine::expr<Arg>& arg)
    {
        return sum(pow2(arg - mean(arg)));
    }

    template <typename Arg>
    inline auto stddev(const engine::expr<Arg>& arg)
    {
        return make_array(sqrt(var(arg)));
    }
}
