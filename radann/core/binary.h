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
        return expr::make_lazy(func::add{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator-(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_lazy(func::sub{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator*(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_lazy(func::mul{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator/(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_lazy(func::div{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto pow(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_lazy(func::pow{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto atan2(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_lazy(func::atan2{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto min(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_lazy(func::min{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto max(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs)
    {
        return expr::make_lazy(func::max{}, lhs, rhs);
    }
}
