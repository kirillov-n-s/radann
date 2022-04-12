#pragma once
#include "engine/unary_map.h"
#include "engine/binary_map.h"
#include "functor/linalg.h"

namespace grad
{
    template <typename Lhs, typename Rhs>
    inline auto dot(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto outer(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto matmul(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Arg>
    inline auto transpose(const engine::expr<Arg>&);

    /*template <typename Arg>
    inline auto inverse(const engine::expr<Arg>&);*/
}

namespace grad
{
    template <typename Lhs, typename Rhs>
    inline auto dot(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        return engine::make_map(functor::dot{}, lhs, rhs).result();
    }

    template <typename Lhs, typename Rhs>
    inline auto outer(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        return engine::make_map(functor::outer{}, lhs, rhs).result();
    }

    template <typename Lhs, typename Rhs>
    inline auto matmul(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        return engine::make_map(functor::matmul{}, lhs, rhs).result();
    }

    template <typename Arg>
    inline auto transpose(const engine::expr<Arg>& arg)
    {
        return engine::make_map(functor::trans{}, arg).result();
    }
}
