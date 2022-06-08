#pragma once
#include "tape_context.h"
#include "grad_unary.h"
#include "grad_binary.h"
#include "is_ad.h"

namespace radann::diff
{
    template<typename Op, typename T, typename Policy, typename Expr>
    void propagate(const expr::access<T, Policy>&, const expr::base<Expr>&);

    template<typename Op, typename Lhs, typename Rhs, typename Expr>
    void propagate(const expr::binary<Op, Lhs, Rhs>&, const expr::base<Expr>&);

    template<typename Op, typename Arg, typename Expr>
    void propagate(const expr::unary<Op, Arg>&, const expr::base<Expr>&);
}

namespace radann::diff
{
    template<typename Op, typename T, typename Policy, typename Expr>
    void propagate(const expr::access<T, Policy> &access, const expr::base<Expr> &mult)
    {
        get_tape<T>()->push_rvalue<Op>(access.grad_index(), mult);
    }

    template<typename Op, typename Lhs, typename Rhs, typename Expr>
    void propagate(const expr::binary<Op, Lhs, Rhs> &binary, const expr::base<Expr> &mult)
    {
        const auto& lhs = binary.lhs();
        const auto& rhs = binary.rhs();
        const auto& op = binary.op();
        if (is_ad(lhs))
            propagate(lhs, grad_lhs(lhs, rhs, mult, op));
        if (is_ad(rhs))
            propagate(rhs, grad_rhs(lhs, rhs, mult, op));
    }

    template<typename Op, typename Arg, typename Expr>
    void propagate(const expr::unary<Op, Arg> &unary, const expr::base<Expr> &mult)
    {
        const auto& arg = unary.arg();
        propagate(arg, grad(arg, mult, unary.op()));
    }
}
