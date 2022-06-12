#pragma once
#include "array.h"
#include "../expr/unary.h"
#include "../expr/binary.h"

namespace radann::core
{
    template<typename Expr>
    inline auto eval(const expr::base<Expr> &);
}

namespace radann::core
{
    template <typename Expr>
    inline auto eval(const expr::base<Expr>& expr)
    {
        if constexpr(Expr::is_expr)
            return array<typename Expr::value_type, typename Expr::strategy_type> { expr };
        else
            return expr.self();
    }

    template<typename Op, typename Arg>
    auto eager(const Op &op, const expr::base<Arg> &arg)
    {
        if constexpr(Op::does_validate)
            op.validate(arg);

        auto arg_array = eval(arg);
        auto res = op(arg_array);

        if constexpr(decltype(res)::does_record)
            res.template record_grad<Op>(expr::get_access(arg_array));

        return res;
    }

    template<typename Op, typename Lhs, typename Rhs>
    auto eager(const Op &op, const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        if constexpr(Op::does_validate)
            op.validate(lhs, rhs);

        auto lhs_array = eval(lhs);
        auto rhs_array = eval(rhs);
        auto res = op(lhs_array, rhs_array);

        if constexpr(decltype(res)::does_record)
            res.template record_grad<Op>(expr::make_expr(op, lhs_array, rhs_array));

        return res;
    }
}
