#pragma once
#include "expr.h"

namespace grad::engine
{
    template<size_t N, bool AD, typename T>
    class access : public expr<access<N, AD, T>>
    {
    public:
        using value_type = T;
        static constexpr size_t rank = N;
        static constexpr bool is_expr = true;
        static constexpr bool is_autodiff = AD;

    private:
        const T* _data;
        size_t _size;
        shape<N> _shape;

        access(const T*, size_t, const shape<N>&);

    public:
        __host__ __device__ inline
        T operator[](size_t i) const;

        auto shape() const;
        auto shape(size_t) const;

        template<typename Expr>
        friend inline auto get_access(const Expr&);
    };
}

namespace grad::engine
{
    template<size_t N, bool AD, typename T>
    access<N, AD, T>::access(const T *data, size_t size, const ::grad::shape<N> &shape)
        : _data(data), _size(size), _shape(shape) {}

    template<size_t N, bool AD, typename T>
    __host__ __device__
    T access<N, AD, T>::operator[](size_t i) const
    {
        return _data[i % _size];
    }

    template<size_t N, bool AD, typename T>
    auto access<N, AD, T>::shape() const
    {
        return _shape;
    }

    template<size_t N, bool AD, typename T>
    auto access<N, AD, T>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<typename Expr>
    inline auto get_access(const Expr &expr)
    {
        if constexpr (Expr::is_expr)
            return expr;
        else
            return access<Expr::rank, Expr::is_autodiff, typename Expr::value_type>
                { expr.data(), expr.size(), expr.shape() };
    }
}
