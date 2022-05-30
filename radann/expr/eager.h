#pragma once
#include "access.h"
#include "../core/eval.h"

namespace radann::expr
{
    template<typename Op, typename Arg>
    auto make_eager(const Op&, const base<Arg>&);

    template <typename Op, typename Lhs, typename Rhs>
    auto make_eager(const Op&, const base<Lhs>&, const base<Rhs>&);
}

namespace radann::expr
{
    template<typename Op, typename Arg>
    auto make_eager(const Op &op, const base<Arg> &arg)
    {
        if constexpr(Op::requires_validation)
            op.validate(arg);

        auto arg_array = eval(arg);
        auto res = op(arg_array);

        if constexpr(Arg::is_autodiff)
        {
            using T = typename decltype(res)::value_type;
            get_access(arg_array).propagate_grad(op.accumulate_grad(arg_array, constant<T>(1)));
            get_tape<T>()->push_lvalue(res.grad_index());
        }

        return res;
    }

    template<typename Op, typename Lhs, typename Rhs>
    auto make_eager(const Op &op, const base<Lhs> &lhs, const base<Rhs> &rhs)
    {
        if constexpr(Op::requires_validation)
            op.validate(lhs, rhs);

        auto lhs_array = eval(lhs);
        auto rhs_array = eval(rhs);
        auto res = op(lhs_array, rhs_array);

        using Res = decltype(res);
        if constexpr(Res::is_autodiff)
        {
            using T = typename Res::value_type;
            if constexpr(Lhs::is_autodiff)
                get_access(lhs_array).propagate_grad(op.accumulate_grad_lhs(lhs_array,
                                                                                 rhs_array,
                                                                                 constant<T>(1)));
            if constexpr(Rhs::is_autodiff)
                get_access(rhs_array).propagate_grad(op.accumulate_grad_rhs(lhs_array,
                                                                                 rhs_array,
                                                                                 constant<T>(1)));
            get_tape<T>()->push_lvalue(res.grad_index());
        }

        return res;
    }
}
