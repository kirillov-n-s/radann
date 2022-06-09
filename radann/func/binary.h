#pragma once
#include "../expr/binary.h"
#include "../oper/binary.h"

namespace radann::func
{
    template <typename Lhs, typename Rhs>
    inline auto operator+(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator-(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator*(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator/(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto pow(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto atan2(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto min(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto max(const expr::base<Lhs>&, const expr::base<Rhs>&);
}

namespace radann::func
{
    template <typename Lhs, typename Rhs>
    inline auto operator+(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(oper::add{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator-(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(oper::sub{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator*(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(oper::mul{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator/(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(oper::div{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto pow(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(oper::pow{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto atan2(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(oper::atan2{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto min(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(oper::min{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto max(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(oper::max{}, lhs, rhs);
    }
}

using radann::func::operator+;
using radann::func::operator-;
using radann::func::operator*;
using radann::func::operator/;
