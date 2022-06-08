#pragma once
#include "array.h"

namespace radann::core
{
    template <typename Policy, typename Expr>
    inline auto eval(const expr::base<Expr>&);
}

namespace radann::core
{
    template <typename Policy, typename Expr>
    inline auto eval(const expr::base<Expr>& expr)
    {
        if constexpr(Expr::is_expr)
            return array<typename Expr::value_type, Policy> { expr };
        else
            return expr.self();
    }
}
