#pragma once
#include "../engine/binary_lazy.h"
#include "../functor/binary.h"

namespace radar
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

namespace radar
{
    template <typename Lhs, typename Rhs>
    inline auto operator+(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_lazy(functor::add{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator-(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_lazy(functor::sub{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator*(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_lazy(functor::mul{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto operator/(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_lazy(functor::div{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto pow(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_lazy(functor::pow{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto atan2(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_lazy(functor::atan2{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto min(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_lazy(functor::min{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto max(const engine::expr<Lhs> &lhs, const engine::expr<Rhs> &rhs)
    {
        return engine::make_lazy(functor::max{}, lhs, rhs);
    }
}
