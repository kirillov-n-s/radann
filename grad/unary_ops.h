#pragma once
#include "ops/unary.h"
#include "ops/unary_functors.h"
#include "ops/unary_functors_ext.h"

namespace grad
{
    template <typename Arg>
    inline auto operator-(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto abs(const ops::expr<Arg>&);

    template <typename Arg>
    inline auto sqrt(const ops::expr<Arg>&);
}

namespace grad
{
    template <typename Arg>
    inline auto operator-(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::neg{}, arg);
    }

    template <typename Arg>
    inline auto abs(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::abs{}, arg);
    }

    template <typename Arg>
    inline auto sqrt(const ops::expr<Arg> &arg)
    {
        return ops::make_unary(ops::sqrt{}, arg);
    }
}
