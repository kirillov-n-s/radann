#pragma once
#include "../core/linalg.h"

namespace radann::func
{
    /*template <typename Lhs, typename Rhs>
    inline auto dot(const expr::base<Lhs>&, const expr::base<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto outer(const expr::base<Lhs>&, const expr::base<Rhs>&);*/

    template <bool LTrans, bool RTrans, typename Lhs, typename Rhs>
    inline auto matmul(const expr::base<Lhs>&, const expr::base<Rhs>&);

    /*template <typename Arg>
    inline auto transpose(const expr::base<Arg>&);*/

    /*template <typename Arg>
    inline auto inverse(const expr::base<Arg>&);*/
}

namespace radann::func
{
    /*template <typename Lhs, typename Rhs>
    inline auto dot(const expr::base<Lhs>& lhs, const expr::base<Rhs>& rhs)
    {
        return expr::make_eager(oper::dot{}, lhs, rhs);
    }

    template <typename Lhs, typename Rhs>
    inline auto outer(const expr::base<Lhs>& lhs, const expr::base<Rhs>& rhs)
    {
        return expr::make_eager(oper::outer{}, lhs, rhs);
    }*/

    template <bool LTrans = false, bool RTrans = false, typename Lhs, typename Rhs>
    inline auto matmul(const expr::base<Lhs>& lhs, const expr::base<Rhs>& rhs)
    {
        return core::eager(core::matmul<LTrans, RTrans>{}, lhs, rhs);
    }

    /*template <typename Arg>
    inline auto transpose(const expr::base<Arg>& arg)
    {
        return expr::make_eager(oper::trans{}, arg);
    }*/
}
