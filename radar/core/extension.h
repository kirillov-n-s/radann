#pragma once
#include "../engine/unary_lazy.h"
#include "../functor/extension.h"

namespace radar
{
    template <typename Arg>
    inline auto sign(const engine::expr<Arg>&);

    template <typename Arg>
    inline auto pow2(const engine::expr<Arg>&);
}

namespace radar
{
    template <typename Arg>
    inline auto sign(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::sgn{}, arg);
    }

    template <typename Arg>
    inline auto pow2(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::pow2{}, arg);
    }
}
