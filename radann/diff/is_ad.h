#pragma once
#include "../expr/access.h"
#include "../expr/binary.h"
#include "../expr/generator.h"
#include "../expr/unary.h"

namespace radann::diff
{
    template<typename T, typename Policy>
    bool is_ad(const expr::access<T, Policy>&);

    template<typename Op, typename Lhs, typename Rhs>
    bool is_ad(const expr::binary<Op, Lhs, Rhs>&);

    template<typename Seq>
    bool is_ad(const expr::generator<Seq>&);

    template<typename Op, typename Arg>
    bool is_ad(const expr::unary<Op, Arg>&);
}

namespace radann::diff
{
    template<typename T, typename Policy>
    bool is_ad(const expr::access<T, Policy> &access)
    {
        return access.ad();
    }

    template<typename Op, typename Lhs, typename Rhs>
    bool is_ad(const expr::binary<Op, Lhs, Rhs> &binary)
    {
        return is_ad(binary.lhs()) || is_ad(binary.rhs());
    }

    template<typename Seq>
    bool is_ad(const expr::generator<Seq> &term)
    {
        return false;
    }

    template<typename Op, typename Arg>
    bool is_ad(const expr::unary<Op, Arg> &unary)
    {
        return is_ad(unary.arg());
    }
}
