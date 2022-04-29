#pragma once
#include "../engine/unary_eager.h"
#include "../engine/binary_eager.h"
#include "../functor/linalg.h"

namespace radann
{
    template <typename Lhs, typename Rhs>
    inline auto dot(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto outer(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto matmul(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <bool LTrans, typename Lhs, typename Rhs>
    inline auto matmul(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <bool LTrans, bool RTrans, typename Lhs, typename Rhs>
    inline auto matmul(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Arg>
    inline auto transpose(const engine::expr<Arg>&);

    /*template <typename Arg>
    inline auto inverse(const engine::expr<Arg>&);*/
}

namespace radann
{
    template <typename Lhs, typename Rhs>
    inline auto dot(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        return engine::make_eager(functor::dot{}, lhs, rhs).result();
    }

    template <typename Lhs, typename Rhs>
    inline auto outer(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        return engine::make_eager(functor::outer{}, lhs, rhs).result();
    }

    template <typename Lhs, typename Rhs>
    inline auto matmul(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        return engine::make_eager(functor::matmul<false, false>{}, lhs, rhs).result();
    }

    template <bool LTrans, typename Lhs, typename Rhs>
    inline auto matmul(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        return engine::make_eager(functor::matmul<LTrans, false>{}, lhs, rhs).result();
    }

    template <bool LTrans, bool RTrans, typename Lhs, typename Rhs>
    inline auto matmul(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        return engine::make_eager(functor::matmul<LTrans, RTrans>{}, lhs, rhs).result();
    }

    template <typename Arg>
    inline auto transpose(const engine::expr<Arg>& arg)
    {
        return engine::make_eager(functor::trans{}, arg).result();
    }
}
