#pragma once
#include "xtmp/unary_ftors_base.h"
#include "xtmp/unary_ftors_ext.h"

namespace grad
{
    template <typename Arg>
    inline auto operator-(const xtmp::expr<Arg>&);

    template <typename Arg>
    inline auto abs(const xtmp::expr<Arg>&);

    template <typename Arg>
    inline auto sqrt(const xtmp::expr<Arg>&);
}

namespace grad
{
    template <typename Arg>
    inline auto operator-(const xtmp::expr<Arg> &arg)
    {
        return xtmp::make_unary_expr(xtmp::neg{}, arg);
    }

    template <typename Arg>
    inline auto abs(const xtmp::expr<Arg> &arg)
    {
        return xtmp::make_unary_expr(xtmp::abs{}, arg);
    }

    template <typename Arg>
    inline auto sqrt(const xtmp::expr<Arg> &arg)
    {
        return xtmp::make_unary_expr(xtmp::sqrt{}, arg);
    }
}
