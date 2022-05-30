#pragma once

namespace radann::expr
{
    template <typename Expr>
    struct base
    {
        inline const Expr &self() const
        {
            return static_cast<const Expr&>(*this);
        }
    };
}
