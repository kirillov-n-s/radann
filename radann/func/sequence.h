#pragma once
#include "../expr/term.h"
#include "../oper/sequence.h"

namespace radann::func
{
    template<typename T>
    inline auto constant(T);

    inline auto operator""_C(long double);

    inline auto operator""_fC(long double);

    template<typename T>
    inline auto arithm(T, T);

    template<typename T>
    inline auto geom(T, T);
}

namespace radann::func
{
    template<typename T>
    inline auto constant(T value)
    {
        return expr::make_expr(oper::constant<T>{value});
    }

    inline auto operator""_C(long double value)
    {
        return constant((double)value);
    }

    inline auto operator""_fC(long double value)
    {
        return constant((float)value);
    }

    template<typename T>
    inline auto arithm(T offset, T step)
    {
        return expr::make_expr(oper::arithm<T>(offset, step));
    }

    template<typename T>
    inline auto geom(T scale, T ratio)
    {
        return expr::make_expr(oper::geom<T>(scale, ratio));
    }
}
