#pragma once
#include "eval.h"

namespace grad::engine
{
    template<typename Op, typename Lhs, typename Rhs>
    inline auto map(const Op&, const expr<Lhs>&, const expr<Rhs>&);

    template<typename Op, typename Arg>
    inline auto map(const Op&, const expr<Arg>&);
}

namespace grad::engine
{
    template<typename Op, typename Lhs, typename Rhs>
    inline auto map(const Op &op, const expr<Lhs> &lhs, const expr<Rhs> &rhs)
    {
        if constexpr(Op::requires_validation)
            op.validate(lhs, rhs);
        return op(eval(lhs), eval(rhs));
    }

    template<typename Op, typename Arg>
    inline auto map(const Op &op, const expr<Arg> &arg)
    {
        if constexpr(Op::requires_validation)
            op.validate(arg);
        return op(eval(arg));
    }
}
