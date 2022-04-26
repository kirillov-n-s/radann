#pragma once
#include "array.h"
#include "sequence.h"
#include "random.h"

namespace radar
{
    template <bool AD = autodiff, typename T, size_t N>
    inline auto make_constant(const shape<N>&, T);
    template <bool AD = autodiff, typename T = real, size_t N>
    inline auto make_ones(const shape<N>&);
    template <bool AD = autodiff, typename T, size_t N>
    inline auto make_arithm(const shape<N>&, T, T);
    template <bool AD = autodiff, typename T, size_t N>
    inline auto make_geom(const shape<N>&, T, T);

    template <bool AD = autodiff, typename T = real, size_t N>
    inline auto make_uniform(const shape<N>&, unsigned int = std::random_device{}());
    template <bool AD = autodiff, typename T = real, size_t N>
    inline auto make_normal(const shape<N>&, unsigned int = std::random_device{}());
}

namespace radar
{
    template <bool AD, typename T, size_t N>
    inline auto make_constant(const shape<N>& shape, T value)
    {
        return array<N, AD, T> { shape, constant(value) };
    }

    template <bool AD, typename T, size_t N>
    inline auto make_ones(const shape<N>& shape)
    {
        return array<N, AD, T> { shape, constant<T>(1) };
    }

    template <bool AD, typename T, size_t N>
    inline auto make_arithm(const shape<N>& shape, T offset, T step)
    {
        return array<N, AD, T> { shape, arithm(offset, step) };
    }

    template <bool AD, typename T, size_t N>
    inline auto make_geom(const shape<N>& shape, T scale, T ratio)
    {
        return array<N, AD, T> { shape, geom(scale, ratio) };
    }

    template <bool AD, typename T, size_t N>
    inline auto make_uniform(const shape<N>& shape, unsigned int seed)
    {
        return array<N, AD, T> { shape, uniform<T>(seed) };
    }

    template <bool AD, typename T, size_t N>
    inline auto make_normal(const shape<N>& shape, unsigned int seed)
    {
        return array<N, AD, T> { shape, normal<T>(seed) };
    }
}
