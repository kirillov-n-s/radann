#pragma once
#include "expr.h"
#include "../meta/meta.h"

namespace grad::engine
{
    template<typename T, size_t N>
    class access : public expr<access<T, N>>
    {
    public:
        using value_type = T;
        static constexpr size_t rank = N;

    private:
        const T* _data;
        size_t _size;
        shape<N> _shape;

        access(const T*, size_t, const shape<N>&);

    public:
        __host__ __device__ inline
        T operator[](size_t i) const;

        auto shape() const;

        template<typename Expr>
        friend inline auto get_access(const Expr&);
    };
}

namespace grad::engine
{
    template<typename T, size_t N>
    access<T, N>::access(const T *data, size_t size, const ::grad::shape<N> &shape)
        : _data(data), _size(size), _shape(shape) {}

    template<typename T, size_t N>
    __host__ __device__
    T access<T, N>::operator[](size_t i) const
    {
        return _data[i % _size];
    }

    template<typename T, size_t N>
    auto access<T, N>::shape() const
    {
        return _shape;
    }

    template<typename Expr>
    inline auto get_access(const Expr &expr)
    {
        if constexpr (meta::has_subscript<Expr>::value)
            return expr;
        else
            return access<typename Expr::value_type, Expr::rank> { expr.data(), expr.size(), expr.shape() };
    }
}
