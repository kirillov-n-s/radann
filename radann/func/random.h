#pragma once
#include "../cuda/random.h"

namespace radann::func
{
    template <typename T>
    class uniform
    {
    public:
        using value_type = T;

    private:
        unsigned int _seed;

    public:
        uniform(unsigned int seed)
            : _seed(seed) {};

        __device__ inline
        T operator()(size_t i) const
        {
            return cuda::random<T>{ _seed, i }.uniform();
        }
    };

    template <typename T>
    class normal
    {
    public:
        using value_type = T;

    private:
        unsigned int _seed;

    public:
        normal(unsigned int seed)
            : _seed(seed) {};

        __device__ inline
        T operator()(size_t i) const
        {
            return cuda::random<T>{ _seed, i }.normal();
        }
    };
}
