#pragma once
#include "../core/util.h"
#include "../core/binary.h"
#include "../core/unary.h"

namespace radann::diff
{
    template<typename Op>
    struct grad_lhs
    {
        static_assert(radann::always_false_v<Op>, "Operator type is not supported.");
    };

    template<typename Op>
    struct grad_rhs
    {
        static_assert(radann::always_false_v<Op>, "Operator type is not supported.");
    };

    template<>
    struct grad_lhs<radann::func::add>
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
    struct grad_rhs<radann::func::add>
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
    struct grad_lhs<radann::func::sub>
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
    struct grad_rhs<radann::func::sub>
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
    struct grad_lhs<radann::func::mul>
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
    struct grad_rhs<radann::func::mul>
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
    struct grad_lhs<radann::func::div>
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
    struct grad_rhs<radann::func::div>
    {
        template<typename Lhs, typename Rhs, typename Mult>
        auto operator()(const expr::base<Lhs> &lhs,
                        const expr::base<Rhs> &rhs,
                        const expr::base<Mult> &mult) const
        {
            return -lhs / radann::pow2(rhs) * mult;
        }
    };
}
