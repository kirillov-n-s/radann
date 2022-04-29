#pragma once
#include "engine/term.h"
#include "functor/sequence.h"

namespace radann
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

namespace radann
{
    template<typename T>
    inline auto constant(T value)
    {
        return engine::make_term(functor::constant<T>{value});
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
        return engine::make_term(functor::arithm<T>(offset, step));
    }

    template<typename T>
    inline auto geom(T scale, T ratio)
    {
        return engine::make_term(functor::geom<T>(scale, ratio));
    }
}
