#pragma once
#include <vector>
#include "backward.h"

namespace radann::diff
{
    template<typename T>
    class tape_context;

    template<typename T>
    class tape
    {
    private:
        std::vector<size_t> _lvalue_indices;
        std::vector<size_t> _last_op_indices;

        std::vector<array_no_ad<T>> _multipliers;
        std::vector<size_t> _rvalue_indices;
        std::vector<backward_function<T>> _backward_functions;

        std::vector<array_no_ad<T>> _gradients;

        size_t _next_index = 0;

        tape() = default;

        size_t push_grad(array_no_ad<T>);

    public:
        tape(const tape&) = delete;
        friend class tape_context<T>;

        size_t create_grad(const core::shape&);
        size_t derive_grad(size_t, const core::shape&, size_t);

        array_no_ad<T> get_grad(size_t) const;
        template<typename Expr>
        void set_grad(size_t, const expr::base<Expr>&);

        template<typename Tag, typename Expr>
        void push_rvalue(size_t, const expr::base<Expr>&);
        void push_lvalue(size_t);

        void reverse();
        void clear();
    };
}

namespace radann::diff
{
    template<typename T>
    size_t tape<T>::push_grad(array_no_ad<T> grad)
    {
        _gradients.push_back(grad);
        return _next_index++;
    }

    template<typename T>
    size_t tape<T>::create_grad(const core::shape &shape)
    {
        return push_grad(array_no_ad<T> { shape });
    }

    template<typename T>
    size_t tape<T>::derive_grad(size_t base_index, const core::shape &shape, size_t offset)
    {
        auto& base = _gradients[base_index];
        return push_grad(array_no_ad<T> { base.storage(), shape, offset, base.grad_index() });
    }

    template<typename T>
    array_no_ad<T> tape<T>::get_grad(size_t index) const
    {
        return _gradients[index];
    }

    template<typename T>
    template<typename Expr>
    void tape<T>::set_grad(size_t index, const expr::base<Expr> &grad)
    {
        _gradients[index] = grad;
    }

    template<typename T>
    template<typename Tag, typename Expr>
    void tape<T>::push_rvalue(size_t index, const expr::base<Expr> &mult)
    {
        array_no_ad<T> array { mult };
        _multipliers.push_back(array);
        _rvalue_indices.push_back(index);
        _backward_functions.push_back(&backward<Tag>::function);
    }

    template<typename T>
    void tape<T>::push_lvalue(size_t index)
    {
        _lvalue_indices.push_back(index);
        _last_op_indices.push_back(_rvalue_indices.size());
    }

    template<typename T>
    void tape<T>::reverse()
    {
        for (size_t i = _lvalue_indices.size() - 1; i > 0; i--)
        {
            array_no_ad<T> output_grad = expr::get_access(_gradients[_lvalue_indices[i]]);
            _gradients[_lvalue_indices[i]] = func::constant<T>(0);
            for (size_t j = _last_op_indices[i - 1]; j < _last_op_indices[i]; j++)
                _backward_functions[j](_gradients[_rvalue_indices[j]],
                                       output_grad,
                                       _multipliers[j]);
        }

        array_no_ad<T> output_grad = expr::get_access(_gradients[_lvalue_indices[0]]);
        _gradients[_lvalue_indices[0]] = func::constant<T>(0);
        for (size_t j = 0; j < _last_op_indices[0]; j++)
            _backward_functions[j](_gradients[_rvalue_indices[j]],
                                   output_grad,
                                   _multipliers[j]);
    }

    template<typename T>
    void tape<T>::clear()
    {
        _lvalue_indices.clear();
        _last_op_indices.clear();
        _rvalue_indices.clear();
        _multipliers.clear();
        _backward_functions.clear();
        //_next_index = 0;
        /*for (auto& grad : _gradients)
            grad = func::constant<T>(0);*/
    }
}
