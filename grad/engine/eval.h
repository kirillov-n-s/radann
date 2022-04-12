#pragma once
#include "../core/array.h"

namespace grad::engine
{
    template <typename Expr>
    inline auto eval(const engine::expr<Expr>&);
}

namespace grad::engine
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
