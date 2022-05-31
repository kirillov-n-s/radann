#pragma once
#include "array.h"
#include "sequence.h"
#include "random.h"

namespace radann
{
    template <typename T>
    inline auto make_constant(const shape&, T, bool = autodiff);
    template <typename T = real>
    inline auto make_ones(const shape&, bool = autodiff);
    template <typename T>
    inline auto make_arithm(const shape&, T, T, bool = autodiff);
    template <typename T>
    inline auto make_geom(const shape&, T, T, bool = autodiff);

    template <typename T = real>
    inline auto make_uniform(const shape&, bool = autodiff, unsigned int = std::random_device{}());
    template <typename T = real>
    inline auto make_normal(const shape&, bool = autodiff, unsigned int = std::random_device{}());
}

namespace radann
{
    template <typename T>
    inline auto make_constant(const shape& shape, T value, bool ad)
    {
        return array<T> { shape, constant(value), ad };
    }

    template <typename T>
    inline auto make_ones(const shape& shape, bool ad)
    {
        return array<T> { shape, constant<T>(1), ad };
    }

    template <typename T>
    inline auto make_arithm(const shape& shape, T offset, T step, bool ad)
    {
        return array<T> { shape, arithm(offset, step), ad };
    }

    template <typename T>
    inline auto make_geom(const shape& shape, T scale, T ratio, bool ad)
    {
        return array<T> { shape, geom(scale, ratio), ad };
    }

    template <typename T>
    inline auto make_uniform(const shape& shape, bool ad, unsigned int seed)
    {
        return array<T> { shape, uniform<T>(seed), ad };
    }

    template <typename T>
    inline auto make_normal(const shape& shape, bool ad, unsigned int seed)
    {
        return array<T> { shape, normal<T>(seed), ad };
    }
}
