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

    template <typename Lhs, typename Rhs>
    inline auto operator^(const ops::expr<Lhs>&, const ops::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto pow(const ops::expr<Lhs>&, const ops::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto atan2(const ops::expr<Lhs>&, const ops::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto min(const ops::expr<Lhs>&, const ops::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto max(const ops::expr<Lhs>&, const ops::expr<Rhs>&);
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

    template <typename Lhs, typename Rhs>
    inline auto operator^(const ops::expr<Lhs> &lhs, const ops::expr<Rhs> &rhs)
    {
        return ops::make_binary(ops::pow{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto pow(const ops::expr<Lhs> &lhs, const ops::expr<Rhs> &rhs)
    {
        return ops::make_binary(ops::pow{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto atan2(const ops::expr<Lhs> &lhs, const ops::expr<Rhs> &rhs)
    {
        return ops::make_binary(ops::atan2{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto min(const ops::expr<Lhs> &lhs, const ops::expr<Rhs> &rhs)
    {
        return ops::make_binary(ops::min{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto max(const ops::expr<Lhs> &lhs, const ops::expr<Rhs> &rhs)
    {
        return ops::make_binary(ops::max{}, lhs, rhs);
    }
}
