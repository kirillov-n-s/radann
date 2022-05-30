#pragma once
#include "../expr/eager.h"
#include "../func/linalg.h"

namespace radann
{
    template <typename Lhs, typename Rhs>
    inline auto dot(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto outer(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto matmul(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <bool LTrans, typename Lhs, typename Rhs>
    inline auto matmul(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <bool LTrans, bool RTrans, typename Lhs, typename Rhs>
    inline auto matmul(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Arg>
    inline auto transpose(const expr::base<Arg>&);

    /*template <typename Arg>
    inline auto inverse(const expr::base<Arg>&);*/
}

namespace radann
{
    template <typename Lhs, typename Rhs>
    inline auto dot(const expr::base<Lhs>& lhs, const expr::base<Rhs>& rhs)
    {
        return expr::make_eager(func::dot{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto outer(const expr::base<Lhs>& lhs, const expr::base<Rhs>& rhs)
    {
        return expr::make_eager(func::outer{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto matmul(const expr::base<Lhs>& lhs, const expr::base<Rhs>& rhs)
    {
        return expr::make_eager(func::matmul<false, false>{}, lhs, rhs);
    }

    template <bool LTrans, typename Lhs, typename Rhs>
    inline auto matmul(const expr::base<Lhs>& lhs, const expr::base<Rhs>& rhs)
    {
        return expr::make_eager(func::matmul<LTrans, false>{}, lhs, rhs);
    }

    template <bool LTrans, bool RTrans, typename Lhs, typename Rhs>
    inline auto matmul(const expr::base<Lhs>& lhs, const expr::base<Rhs>& rhs)
    {
        return expr::make_eager(func::matmul<LTrans, RTrans>{}, lhs, rhs);
    }

    template <typename Arg>
    inline auto transpose(const expr::base<Arg>& arg)
    {
        return expr::make_eager(func::trans{}, arg);
    }
}
