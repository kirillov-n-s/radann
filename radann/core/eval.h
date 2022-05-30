#pragma once
#include "array.h"

namespace radann
{
    template <typename Expr>
    inline auto eval(const expr::base<Expr>&);
}

namespace radann
{
    template <typename Expr>
    inline auto eval(const expr::base<Expr>& expr)
    {
        if constexpr(Expr::is_expr)
            return make_array(expr);
        else
            return expr.self();
    }
}
