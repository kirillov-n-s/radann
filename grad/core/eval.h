#pragma once
#include "array.h"

namespace grad
{
    template <typename Expr>
    inline auto eval(const engine::expr<Expr>&);
}

namespace grad
{
    template <typename Expr>
    inline auto eval(const engine::expr<Expr>& expr)
    {
        if constexpr(Expr::is_expr)
            return make_array(expr);
        else
            return expr.self();
    }
}
