#pragma once

namespace radann::engine
{
    template <typename Expr>
    struct expr
    {
        inline const Expr &self() const
        {
            return static_cast<const Expr&>(*this);
        }
    };
}