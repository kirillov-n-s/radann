#pragma once
#include "../func/binary.h"
#include "../func/unary.h"

namespace radann::diff
{
    template<typename Lhs, typename Rhs, typename Mult>
    auto grad_lhs(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs, const expr::base<Mult> &mult,
                  const oper::add&)
    {
        return mult.self();
    }

    template<typename Lhs, typename Rhs, typename Mult>
    auto grad_rhs(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs, const expr::base<Mult> &mult,
                  const oper::add&)
    {
        return mult.self();
    }

    template<typename Lhs, typename Rhs, typename Mult>
    auto grad_lhs(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs, const expr::base<Mult> &mult,
                  const oper::sub&)
    {
        return mult.self();
    }

    template<typename Lhs, typename Rhs, typename Mult>
    auto grad_rhs(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs, const expr::base<Mult> &mult,
                  const oper::sub&)
    {
        return -mult;
    }

    template<typename Lhs, typename Rhs, typename Mult>
    auto grad_lhs(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs, const expr::base<Mult> &mult,
                  const oper::mul&)
    {
        return rhs * mult;
    }

    template<typename Lhs, typename Rhs, typename Mult>
    auto grad_rhs(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs, const expr::base<Mult> &mult,
                  const oper::mul&)
    {
        return lhs * mult;
    }

    template<typename Lhs, typename Rhs, typename Mult>
    auto grad_lhs(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs, const expr::base<Mult> &mult,
                  const oper::div&)
    {
        return mult / rhs;
    }

    template<typename Lhs, typename Rhs, typename Mult>
    auto grad_rhs(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs, const expr::base<Mult> &mult,
                  const oper::div&)
    {
        return -lhs / func::pow2(rhs) * mult;
    }
}
