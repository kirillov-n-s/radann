#pragma once
#include "core/array.h"
#include "sequence.h"
#include "random.h"

namespace grad
{
    template <typename T, size_t N>
    inline auto make_constant(const shape<N>&, T);
    template <typename T, size_t N>
    inline auto make_ones(const shape<N>&);
    template <typename T, size_t N>
    inline auto make_arithm(const shape<N>&, T, T);
    template <typename T, size_t N>
    inline auto make_geom(const shape<N>&, T, T);

    template <typename T, size_t N>
    inline auto make_uniform(const shape<N>&, unsigned int = std::random_device{}());
    template <typename T, size_t N>
    inline auto make_normal(const shape<N>&, unsigned int = std::random_device{}());
}

namespace grad
{
    template <typename T, size_t N>
    inline auto make_constant(const shape<N>& shape, T value)
    {
        return array<T, shape.rank> { shape, constant(value) };
    }

    template <typename T, size_t N>
    inline auto make_ones(const shape<N>& shape)
    {
        return array<T, shape.rank> { shape, constant<T>(1) };
    }

    template <typename T, size_t N>
    inline auto make_arithm(const shape<N>& shape, T offset, T step)
    {
        return array<T, shape.rank> { shape, arithm(offset, step) };
    }

    template <typename T, size_t N>
    inline auto make_geom(const shape<N>& shape, T scale, T ratio)
    {
        return array<T, shape.rank> { shape, geom(scale, ratio) };
    }

    template <typename T, size_t N>
    inline auto make_uniform(const shape<N>& shape, unsigned int seed)
    {
        return array<T, shape.rank> { shape, uniform<T>(seed) };
    }

    template <typename T, size_t N>
    inline auto make_normal(const shape<N>& shape, unsigned int seed)
    {
        return array<T, shape.rank> { shape, normal<T>(seed) };
    }
}
