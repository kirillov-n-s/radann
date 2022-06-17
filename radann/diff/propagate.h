#pragma once
#include "tape_context.h"
#include "grad_unary.h"
#include "grad_binary.h"
#include "is_ad.h"

namespace radann::diff
{
    template<typename Tag, typename T, typename Strategy, typename Expr>
    void propagate(const expr::access<T, Strategy>&, const expr::base<Expr>&);

    template<typename Tag, typename Op, typename Lhs, typename Rhs, typename Expr>
    void propagate(const expr::binary<Op, Lhs, Rhs>&, const expr::base<Expr>&);

    template<typename Tag, typename Seq, typename Expr>
    void propagate(const expr::generator<Seq>&, const expr::base<Expr>&);

    template<typename Tag, typename Op, typename Arg, typename Expr>
    void propagate(const expr::unary<Op, Arg>&, const expr::base<Expr>&);
}

namespace radann::diff
{
    template<typename Tag, typename T, typename Strategy, typename Expr>
    void propagate(const expr::access<T, Strategy> &access, const expr::base<Expr> &mult)
    {
        get_tape<T>()->template push_term<Tag>(access.grad_index(), mult);
    }

    template<typename Tag, typename Op, typename Lhs, typename Rhs, typename Expr>
    void propagate(const expr::binary<Op, Lhs, Rhs> &binary, const expr::base<Expr> &mult)
    {
        const auto& lhs = binary.lhs();
        const auto& rhs = binary.rhs();
        const auto& op = binary.op();
        if (is_ad(lhs))
            propagate<typename Tag::backward_lhs>(lhs, grad_lhs(lhs, rhs, mult, op));
        if (is_ad(rhs))
            propagate<typename Tag::backward_rhs>(rhs, grad_rhs(lhs, rhs, mult, op));
    }

    template<typename Tag, typename Seq, typename Expr>
    void propagate(const expr::generator<Seq>&, const expr::base<Expr>&) {}

    template<typename Tag, typename Op, typename Arg, typename Expr>
    void propagate(const expr::unary<Op, Arg> &unary, const expr::base<Expr> &mult)
    {
        const auto& arg = unary.arg();
        propagate<Tag>(arg, grad(arg, mult, unary.op()));
    }
}
