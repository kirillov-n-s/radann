#pragma once
#include "../meta/meta.h"
#include "../func/binary.h"
#include "../func/unary.h"

namespace radann::diff
{
    template<typename Op>
    struct grad_lhs
    {
        static_assert(meta::always_false_v<Op>, "Operator type is not supported.");
    };

    template<typename Op>
    struct grad_rhs
    {
        static_assert(meta::always_false_v<Op>, "Operator type is not supported.");
    };

    template<>
    struct grad_lhs<oper::add>
    {
        template<typename Lhs, typename Rhs, typename Mult>
        auto operator()(const expr::base<Lhs> &lhs,
                                 const expr::base<Rhs> &rhs,
                                 const expr::base<Mult> &mult) const
        {
            return mult.self();
        }
    };

    template<>
    struct grad_rhs<oper::add>
    {
        template<typename Lhs, typename Rhs, typename Mult>
        auto operator()(const expr::base<Lhs> &lhs,
                        const expr::base<Rhs> &rhs,
                        const expr::base<Mult> &mult) const
        {
            return mult.self();
        }
    };

    template<>
    struct grad_lhs<oper::sub>
    {
        template<typename Lhs, typename Rhs, typename Mult>
        auto operator()(const expr::base<Lhs> &lhs,
                        const expr::base<Rhs> &rhs,
                        const expr::base<Mult> &mult) const
        {
            return mult.self();
        }
    };

    template<>
    struct grad_rhs<oper::sub>
    {
        template<typename Lhs, typename Rhs, typename Mult>
        auto operator()(const expr::base<Lhs> &lhs,
                        const expr::base<Rhs> &rhs,
                        const expr::base<Mult> &mult) const
        {
            return -mult;
        }
    };

    template<>
    struct grad_lhs<oper::mul>
    {
        template<typename Lhs, typename Rhs, typename Mult>
        auto operator()(const expr::base<Lhs> &lhs,
                        const expr::base<Rhs> &rhs,
                        const expr::base<Mult> &mult) const
        {
            return rhs * mult;
        }
    };

    template<>
    struct grad_rhs<oper::mul>
    {
        template<typename Lhs, typename Rhs, typename Mult>
        auto operator()(const expr::base<Lhs> &lhs,
                        const expr::base<Rhs> &rhs,
                        const expr::base<Mult> &mult) const
        {
            return lhs * mult;
        }
    };

    template<>
    struct grad_lhs<oper::div>
    {
        template<typename Lhs, typename Rhs, typename Mult>
        auto operator()(const expr::base<Lhs> &lhs,
                        const expr::base<Rhs> &rhs,
                        const expr::base<Mult> &mult) const
        {
            return mult / rhs;
        }
    };

    template<>
    struct grad_rhs<oper::div>
    {
        template<typename Lhs, typename Rhs, typename Mult>
        auto operator()(const expr::base<Lhs> &lhs,
                        const expr::base<Rhs> &rhs,
                        const expr::base<Mult> &mult) const
        {
            return -lhs / func::pow2(rhs) * mult;
        }
    };
}
