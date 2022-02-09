#pragma once
#include <type_traits>
#include "expr.h"

namespace grad::ops
{
    template<typename T, size_t N>
    class access : public expr<access<T, N>>
    {
    public:
        using value_type = T;
        static const size_t rank = N;

    private:
        const T* _data;
        size_t _size;
        shape<N> _shape;

        access(const T*, size_t, const shape<N>&);

    public:
        __host__ __device__ inline
        T operator[](size_t i) const;

        shape<N> shape() const;

        /*template<typename T, size_t N>
        friend inline access<T, N> get_access(const array<T, N>&);*/

        template<typename Expr>
        friend inline auto get_access(const Expr&);
    };

    template<typename Expr, typename = void>
    struct has_subscript : std::false_type {};

    template<typename Expr>
    struct has_subscript<Expr,
        std::enable_if_t<std::is_same_v<decltype(&Expr::operator[]),
            typename Expr::value_type (Expr::*)(size_t) const>>>
                : std::true_type {};

    /*template<typename Expr, std::enable_if_t<has_subscript<Expr>::value, bool> = true>
    inline Expr get_access(const Expr&);*/
}

namespace grad::ops
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
    shape<N> access<T, N>::shape() const
    {
        return _shape;
    }

    template<typename Expr>
    inline auto get_access(const Expr &expr)
    {
        if constexpr (has_subscript<Expr>::value)
            return expr;
        else
            return access<typename Expr::value_type, Expr::rank> { expr.data(), expr.size(), expr.shape() };
    }

    /*template<typename T, size_t N>
    inline access<T, N> get_access(const array<T, N> &array)
    {
        return { array.data(), array.size(), array.shape() };
    }*/

    /*template<typename Expr, std::enable_if_t<has_subscript<Expr>::value, bool>>
    inline Expr get_access(const Expr &expr)
    {
        return expr;
    }*/
}
