#pragma once
#include "engine/unary_eltwise.h"
#include "functor/activation.h"

namespace grad
{
    template <typename Arg>
    inline auto sigmoid(const engine::expr<Arg>&);
}

namespace grad
{
    template <typename Arg>
    inline auto sigmoid(const engine::expr<Arg> &arg)
    {
        return engine::make_eltwise(functor::sigmoid{}, arg);
    }
}
