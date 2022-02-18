#pragma once
#include <type_traits>
#include "curand_kernel.h"

namespace grad::cuda
{
    template <typename T>
    inline constexpr bool always_false_v = false;

    template <typename T>
    class random
    {
    private:
        curandState_t _state;

    public:
        __device__ random(unsigned int, size_t);

        __device__ T uniform();
        __device__ T normal();
    };
}

namespace grad::cuda
{
    template <typename T>
    __device__ random<T>::random(unsigned int seed, size_t idx)
    {
        curand_init(seed, idx, 0, &_state);
    }

    template <typename T>
    __device__ T random<T>::uniform()
    {
        if constexpr(std::is_same_v<T, float>)
            return curand_uniform(&_state);
        else if constexpr(std::is_same_v<T, double>)
            return curand_uniform_double(&_state);
        else
            static_assert(always_false_v<T>, "PRNG for that type is not specialized.");
    }

    template <typename T>
    __device__ T random<T>::normal()
    {
        if constexpr(std::is_same_v<T, float>)
            return curand_normal(&_state);
        else if constexpr(std::is_same_v<T, double>)
            return curand_normal_double(&_state);
        else
            static_assert(always_false_v<T>, "PRNG for that type is not specialized.");
    }
}
