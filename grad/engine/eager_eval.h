#pragma once
#include "../creation.h"

namespace grad::engine
{
    template <typename Expr>
    inline auto eager_eval(const engine::expr<Expr>&);
}

namespace grad::engine
{
    template <typename Expr>
    inline auto eager_eval(const engine::expr<Expr>& expr)
    {
        if constexpr(Expr::is_expr)
            return make_array(expr);
        else
            return expr.self();
    }
}
