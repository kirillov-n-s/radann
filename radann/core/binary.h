#pragma once
#include "../expr/binary.h"
#include "../func/binary.h"

namespace radann
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

namespace radann
{
    template <typename Lhs, typename Rhs>
    inline auto operator+(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(func::add{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator-(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(func::sub{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator*(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(func::mul{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator/(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(func::div{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto pow(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(func::pow{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto atan2(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(func::atan2{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto min(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(func::min{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto max(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_expr(func::max{}, lhs, rhs);
    }
}
