#pragma once
#include "expr.h"
#include "tape_context.h"

namespace radar::engine
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
        size_t _grad_index;

    public:
        access(const T*, size_t, const shape<N>&, size_t);

        __host__ __device__ inline
        T operator[](size_t i) const;

        auto shape() const;
        size_t shape(size_t) const;

        template<typename Expr>
        void propagate_grad(const expr<Expr>&) const;
    };

    template<size_t N, typename T>
    class access<N, false, T> : public expr<access<N, false, T>>
    {
    public:
        using value_type = T;
        static constexpr size_t rank = N;
        static constexpr bool is_expr = true;
        static constexpr bool is_autodiff = false;

    private:
        const T* _data;
        size_t _size;
        shape<N> _shape;

    public:
        access(const T*, size_t, const shape<N>&);

        __host__ __device__ inline
        T operator[](size_t i) const;

        auto shape() const;
        size_t shape(size_t) const;
    };

    template<typename Expr>
    inline auto get_access(const Expr&);
}

namespace radar::engine
{
    template<size_t N, bool AD, typename T>
    access<N, AD, T>::access(const T *data, size_t size, const radar::shape<N> &shape, size_t grad_index)
        : _data(data), _size(size), _shape(shape), _grad_index(grad_index)
    {}

    template<size_t N, typename T>
    access<N, false, T>::access(const T *data, size_t size, const radar::shape<N> &shape)
        : _data(data), _size(size), _shape(shape)
    {}

    template<size_t N, bool AD, typename T>
    __host__ __device__
    T access<N, AD, T>::operator[](size_t i) const
    {
        return _data[i % _size];
    }

    template<size_t N, typename T>
    __host__ __device__
    T access<N, false, T>::operator[](size_t i) const
    {
        return _data[i % _size];
    }

    template<size_t N, bool AD, typename T>
    auto access<N, AD, T>::shape() const
    {
        return _shape;
    }

    template<size_t N, typename T>
    auto access<N, false, T>::shape() const
    {
        return _shape;
    }

    template<size_t N, bool AD, typename T>
    size_t access<N, AD, T>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<size_t N, typename T>
    size_t access<N, false, T>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    void access<N, AD, T>::propagate_grad(const expr<Expr> &mult) const
    {
        auto mult_self = mult.self();
        auto mult_eval = new cuda::unique_array<T> { _size };
        cuda::assign(mult_eval->data(), mult_eval->size(), mult_self);
        get_tape<T>()->push_rvalue(mult_eval, _grad_index);
    }

    template<typename Expr>
    inline auto get_access(const Expr &expr)
    {
        if constexpr(Expr::is_expr)
            return expr;
        else
        {
            if constexpr(Expr::is_autodiff)
                return access<Expr::rank, true, typename Expr::value_type>
                        {expr.data(), expr.size(), expr.shape(), expr.grad_index()};
            else
                return access<Expr::rank, false, typename Expr::value_type>
                        {expr.data(), expr.size(), expr.shape()};
        }
    }
}
