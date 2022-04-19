#pragma once
#include "core/array.h"
#include "sequence.h"
#include "random.h"

namespace grad
{
    template <typename T, size_t N>
    inline auto make_constant(const shape<N>&, T);
    template <typename T = real, size_t N>
    inline auto make_ones(const shape<N>&);
    template <typename T, size_t N>
    inline auto make_arithm(const shape<N>&, T, T);
    template <typename T, size_t N>
    inline auto make_geom(const shape<N>&, T, T);

    template <typename T = real, size_t N>
    inline auto make_uniform(const shape<N>&, unsigned int = std::random_device{}());
    template <typename T = real, size_t N>
    inline auto make_normal(const shape<N>&, unsigned int = std::random_device{}());
}

namespace grad
{
    template <typename T, size_t N>
    inline auto make_constant(const shape<N>& shape, T value)
    {
        return array<N, T> { shape, constant(value) };
    }

    template <typename T, size_t N>
    inline auto make_ones(const shape<N>& shape)
    {
        return array<N, T> { shape, constant<T>(1) };
    }

    template <typename T, size_t N>
    inline auto make_arithm(const shape<N>& shape, T offset, T step)
    {
        return array<N, T> { shape, arithm(offset, step) };
    }

    template <typename T, size_t N>
    inline auto make_geom(const shape<N>& shape, T scale, T ratio)
    {
        return array<N, T> { shape, geom(scale, ratio) };
    }

    template <typename T, size_t N>
    inline auto make_uniform(const shape<N>& shape, unsigned int seed)
    {
        return array<N, T> { shape, uniform<T>(seed) };
    }

    template <typename T, size_t N>
    inline auto make_normal(const shape<N>& shape, unsigned int seed)
    {
        return array<N, T> { shape, normal<T>(seed) };
    }
}
