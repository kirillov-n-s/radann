#pragma once
#include <type_traits>

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

    template <typename T1, typename T2>
    inline auto div_ceil(T1, T2);

    template <typename T>
    inline T half_ceil(T);
}

namespace grad::util
{
    template <typename T1, typename T2>
    inline auto div_ceil(T1 x, T2 y)
    {
        return (std::common_type_t<T1, T2>)::ceil((double)x / y);
    }

    template <typename T>
    inline T half_ceil(T x)
    {
        return (T)::ceil(x * 0.5);
    }
}
