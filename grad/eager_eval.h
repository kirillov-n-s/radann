#pragma once
#include "creation.h"

namespace grad
{
    template <typename Expr>
    inline auto eager_eval(const engine::expr<Expr>&);
}

namespace grad
{
    template <typename Expr>
    inline auto eager_eval(const engine::expr<Expr>& expr)
    {
        if constexpr(Expr::is_expr)
            return make_array<Expr::value_type>(expr);
        else
            return expr.self();
    }
}
