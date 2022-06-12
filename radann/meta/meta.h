#pragma once

namespace radann::meta
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

    template<typename T, typename U>
    struct same_type {};

    template<typename T>
    struct same_type<T, T>
    {
        using type = T;
    };

    struct any_type {};

    template<typename T>
    struct same_type<any_type, T>
    {
        using type = T;
    };

    template<typename T>
    struct same_type<T, any_type>
    {
        using type = T;
    };

    template<>
    struct same_type<any_type, any_type>
    {
        using type = any_type;
    };

    template<typename T, typename U>
    using same_type_t = typename same_type<T, U>::type;
}
