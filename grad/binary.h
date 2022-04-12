#pragma once
#include "engine/binary_eltwise.h"
#include "functor/binary.h"

namespace grad
{
    template <typename Lhs, typename Rhs>
    inline auto operator+(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator-(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator*(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto operator/(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto pow(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto atan2(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto min(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto max(const engine::expr<Lhs>&, const engine::expr<Rhs>&);
}

namespace grad
{
    template <typename Lhs, typename Rhs>
    inline auto operator+(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_eltwise(functor::add{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator-(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_eltwise(functor::sub{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator*(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_eltwise(functor::mul{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator/(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_eltwise(functor::div{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto pow(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_eltwise(functor::pow{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto atan2(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_eltwise(functor::atan2{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto min(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_eltwise(functor::min{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto max(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_eltwise(functor::max{}, lhs, rhs);
    }
}
