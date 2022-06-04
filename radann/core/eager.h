#pragma once
#include "../expr/access.h"
#include "eval.h"

namespace radann::core
{
    template<typename Op, typename Arg>
    auto eager(const Op&, const expr::base<Arg>&);

    template <typename Op, typename Lhs, typename Rhs>
    auto eager(const Op&, const expr::base<Lhs>&, const expr::base<Rhs>&);
}

namespace radann::core
{
    template<typename Op, typename Arg>
    auto eager(const Op &op, const expr::base<Arg> &arg)
    {
        if constexpr(Op::requires_validation)
            op.validate(arg);

        auto arg_array = eval(arg);
        auto res = op(arg_array);

        if (res.ad())
        {
            using T = typename decltype(res)::value_type;
            get_access(arg_array).propagate_grad(op.accumulate_grad(arg_array, constant<T>(1)));
            get_tape<T>()->push_lvalue(res.grad_index());
        }

        return res;
    }

    template<typename Op, typename Lhs, typename Rhs>
    auto eager(const Op &op, const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        if constexpr(Op::requires_validation)
            op.validate(lhs, rhs);

        auto lhs_array = eval(lhs);
        auto rhs_array = eval(rhs);
        auto res = op(lhs_array, rhs_array);

        if (res.ad())
        {
            using T = typename decltype(res)::value_type;
            if (lhs.ad())
                get_access(lhs_array).propagate_grad(op.accumulate_grad_lhs(lhs_array, rhs_array));
            if (rhs.ad())
                get_access(rhs_array).propagate_grad(op.accumulate_grad_rhs(lhs_array, rhs_array));
            get_tape<T>()->push_lvalue(res.grad_index());
        }

        return res;
    }
}
