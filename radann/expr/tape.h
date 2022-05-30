#pragma once
#include <vector>
#include <list>
#include "../cuda/shared_array.h"
#include "../cuda/unique_array.h"
#include "../cuda/assign.h"

namespace radann::expr
{
    template<typename T>
    class tape_context;

    template<typename T>
    class tape
    {
    private:
        std::vector<size_t> _lvalue_indices;
        std::vector<size_t> _last_op_indices;

        std::vector<cuda::unique_array<T>*> _multipliers;
        std::vector<size_t> _rvalue_indices;

        std::vector<cuda::shared_array<T>*> _gradients;

        size_t _next_index = 0;

        tape() = default;

        size_t push_grad(cuda::shared_array<T>*);

    public:
        tape(const tape&) = delete;
        friend class tape_context<T>;

        size_t create_grad(size_t);
        size_t derive_grad(size_t, size_t, size_t);

        const T* get_grad(size_t) const;
        template<typename Expr>
        void set_grad(size_t, const base<Expr>&);

        template<typename Expr>
        void push_rvalue(size_t, const base<Expr>&);
        void push_lvalue(size_t);

        void reverse();
        void clear();
    };
}

namespace radann::expr
{
    template<typename T>
    size_t tape<T>::push_grad(cuda::shared_array<T> *grad)
    {
        _gradients.push_back(grad);
        return _next_index++;
    }

    template<typename T>
    size_t tape<T>::create_grad(size_t size)
    {
        return push_grad(new cuda::shared_array<T>(size));
    }

    template<typename T>
    size_t tape<T>::derive_grad(size_t base_index, size_t size, size_t offset)
    {
        return push_grad(new cuda::shared_array<T>(_gradients[base_index]->storage(), size, offset));
    }

    template<typename T>
    const T *tape<T>::get_grad(size_t index) const
    {
        return _gradients[index]->data();
    }

    template<typename T>
    template<typename Expr>
    void tape<T>::set_grad(size_t index, const base<Expr> &grad)
    {
        auto grad_array = _gradients[index];
        cuda::assign(grad_array->data(), grad_array->size(), grad.self());
    }

    template<typename T>
    template<typename Expr>
    void tape<T>::push_rvalue(size_t index, const base<Expr> &mult)
    {
        auto size = _gradients[index]->size();
        auto array = new cuda::unique_array<T> { size };
        cuda::assign(array->data(), size, mult.self());

        _multipliers.push_back(array);
        _rvalue_indices.push_back(index);
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
            auto output_grad = _gradients[_lvalue_indices[i]];
            for (size_t j = _last_op_indices[i - 1]; j < _last_op_indices[i]; j++)
            {
                auto input_grad = _gradients[_rvalue_indices[j]];
                cuda::reverse_grad(input_grad->data(),
                                   _multipliers[j]->data(),
                                   output_grad->data(),
                                   input_grad->size(),
                                   output_grad->size());
            }
        }

        auto output_grad = _gradients[_lvalue_indices[0]];
        for (size_t j = 0; j < _last_op_indices[0]; j++)
        {
            auto input_grad = _gradients[_rvalue_indices[j]];
            cuda::reverse_grad(input_grad->data(),
                               _multipliers[j]->data(),
                               output_grad->data(),
                               input_grad->size(),
                               output_grad->size());
        }
    }

    template<typename T>
    void tape<T>::clear()
    {
        for (auto& grad : _gradients)
            delete grad;
        for (auto& mult : _multipliers)
            delete mult;
    }
}
