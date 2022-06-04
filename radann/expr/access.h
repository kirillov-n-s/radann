#pragma once
#include "base.h"

namespace radann::expr
{
    template<typename T>
    class access : public base<access<T>>
    {
    public:
        using value_type = T;
        static constexpr bool is_expr = true;

    private:
        const T* _data;
        size_t _size;
        core::shape _shape;
        std::optional<size_t> _grad_index;

        access(const T*, size_t, const core::shape&, const std::optional<size_t>&);

    public:
        __host__ __device__ inline
        T operator[](size_t i) const;

        size_t rank() const;
        auto shape() const;
        size_t shape(size_t) const;

        const std::optional<size_t>& grad_index() const;

        template<typename Expr>
        friend inline auto get_access(const Expr&);
    };
}

namespace radann::expr
{
    template<typename T>
    access<T>::access(const T *data, size_t size, const core::shape &shape, const std::optional<size_t> &grad_index)
        : _data(data), _size(size), _shape(shape), _grad_index(grad_index)
    {}
    
    template<typename T>
    __host__ __device__
    T access<T>::operator[](size_t i) const
    {
        return _data[i % _size];
    }

    template<typename T>
    size_t access<T>::rank() const
    {
        return _shape.rank();
    }

    template<typename T>
    auto access<T>::shape() const
    {
        return _shape;
    }

    template<typename T>
    size_t access<T>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<typename T>
    const std::optional<size_t> &access<T>::grad_index() const
    {
        return _grad_index;
    }

    template<typename Expr>
    inline auto get_access(const Expr &expr)
    {
        if constexpr(Expr::is_expr)
            return expr;
        else
            return access<typename Expr::value_type>
                { expr.data(), expr.size(), expr.shape(), expr.grad_index() };
    }
}
