#pragma once
#include <optional>
#include "base.h"
#include "../core/shape.h"

namespace radann::expr
{
    template<typename T, typename Policy>
    class access :
        public base<access<T, Policy>>,
        public Policy::entry_type
    {
    public:
        using value_type = T;
        using policy_type = Policy;
        static constexpr bool is_expr = true;

    private:
        const T* _data;
        size_t _size;
        core::shape _shape;

        access(const T*, size_t, const core::shape&, const typename Policy::index_type&);

    public:
        __host__ __device__ inline
        T operator[](size_t) const;

        size_t rank() const;
        auto shape() const;
        size_t shape(size_t) const;

        template<typename Expr>
        friend inline auto get_access(const Expr&);
    };
}

namespace radann::expr
{
    template<typename T, typename Policy>
    access<T, Policy>::access(const T *data, size_t size, const core::shape &shape,
                              const typename Policy::index_type &grad_index)
        : Policy::entry_type(grad_index),
          _data(data), _size(size), _shape(shape)
    {}
    
    template<typename T, typename Policy>
    __host__ __device__
    T access<T, Policy>::operator[](size_t i) const
    {
        return _data[i % _size];
    }

    template<typename T, typename Policy>
    size_t access<T, Policy>::rank() const
    {
        return _shape.rank();
    }

    template<typename T, typename Policy>
    auto access<T, Policy>::shape() const
    {
        return _shape;
    }

    template<typename T, typename Policy>
    size_t access<T, Policy>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<typename Expr>
    inline auto get_access(const Expr &expr)
    {
        if constexpr(Expr::is_expr)
            return expr;
        else
            return access<typename Expr::value_type, typename Expr::policy_type>
                { expr.data(), expr.size(), expr.shape(), expr.grad_index() };
    }
}
