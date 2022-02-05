#pragma once
#include "xtmp/binary_ftors_base.h"

namespace grad
{
    template <typename Lhs, typename Rhs>
    inline auto operator+(const xtmp::expr<Lhs>&, const xtmp::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator-(const xtmp::expr<Lhs>&, const xtmp::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator*(const xtmp::expr<Lhs>&, const xtmp::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator/(const xtmp::expr<Lhs>&, const xtmp::expr<Rhs>&);
}

namespace grad
{
    template <typename Lhs, typename Rhs>
    inline auto operator+(const xtmp::expr<Lhs> &lhs, const xtmp::expr<Rhs> &rhs)
    {
        return xtmp::make_binary_expr(xtmp::add{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator-(const xtmp::expr<Lhs> &lhs, const xtmp::expr<Rhs> &rhs)
    {
        return xtmp::make_binary_expr(xtmp::sub{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator*(const xtmp::expr<Lhs> &lhs, const xtmp::expr<Rhs> &rhs)
    {
        return xtmp::make_binary_expr(xtmp::mul{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator/(const xtmp::expr<Lhs> &lhs, const xtmp::expr<Rhs> &rhs)
    {
        return xtmp::make_binary_expr(xtmp::div{}, lhs, rhs);
    }
}
