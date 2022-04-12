#pragma once

namespace grad::util
{
    /*template<typename Expr, typename = void>
    struct has_subscript : std::false_type {};

    template<typename Expr>
    struct has_subscript<Expr,
            std::enable_if_t<std::is_same_v<decltype(&Expr::operator[]),
                    typename Expr::value_type (Expr::*)(size_t) const>>>
        : std::true_type {};*/

    template <typename T>
    inline constexpr bool always_false_v = false;
}
