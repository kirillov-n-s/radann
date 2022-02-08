#pragma once
#include "ops/expr.h"

namespace grad
{
    template<typename T>
    class constant : public ops::expr<constant<T>>
    {
    public:
        using value_type = T;
        static const size_t rank = 0;

    private:
        T _value;

    public:
        constant(T);

        __host__ __device__ inline
        T operator[](size_t i) const;

        shape<0> shape() const;
    };

    template<typename T>
    inline constant<T> make_constant(T);
}

namespace grad
{
    template<typename T>
    constant<T>::constant(T value)
        : _value(value) {}

    template<typename T>
    __host__ __device__
    T constant<T>::operator[](size_t i) const
    {
        return _value;
    }

    template<typename T>
    shape<0> constant<T>::shape() const
    {
        return make_shape();
    }

    template<typename T>
    inline constant<T> make_constant(T value)
    {
        return { value };
    }
}
