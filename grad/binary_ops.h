#pragma once
#include "ops/binary.h"
#include "ops/binary_functors.h"

namespace grad
{
    template <typename Lhs, typename Rhs>
    inline auto operator+(const ops::expr<Lhs>&, const ops::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator-(const ops::expr<Lhs>&, const ops::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator*(const ops::expr<Lhs>&, const ops::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator/(const ops::expr<Lhs>&, const ops::expr<Rhs>&);
}

namespace grad
{
    template <typename Lhs, typename Rhs>
    inline auto operator+(const ops::expr<Lhs> &lhs, const ops::expr<Rhs> &rhs)
    {
        return ops::make_binary(ops::add{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator-(const ops::expr<Lhs> &lhs, const ops::expr<Rhs> &rhs)
    {
        return ops::make_binary(ops::sub{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator*(const ops::expr<Lhs> &lhs, const ops::expr<Rhs> &rhs)
    {
        return ops::make_binary(ops::mul{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator/(const ops::expr<Lhs> &lhs, const ops::expr<Rhs> &rhs)
    {
        return ops::make_binary(ops::div{}, lhs, rhs);
    }
}
