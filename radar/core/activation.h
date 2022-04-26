#pragma once
#include "../engine/unary_lazy.h"
#include "../functor/activation.h"

namespace radar
{
    template <typename Arg>
    inline auto sigmoid(const engine::expr<Arg>&);
}

namespace radar
{
    template <typename Arg>
    inline auto sigmoid(const engine::expr<Arg> &arg)
    {
        return engine::make_lazy(functor::sigmoid{}, arg);
    }
}
