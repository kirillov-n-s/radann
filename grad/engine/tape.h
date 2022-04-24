#pragma once
#include <vector>
#include <list>
#include "../cuda/shared_array.h"
#include "../cuda/unique_array.h"

namespace grad::engine
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
        std::list<size_t> _gaps;

        size_t _next_index = 0;

        tape() = default;

        size_t push_grad(cuda::shared_array<T>*);

    public:
        tape(const tape&) = delete;
        friend class tape_context<T>;

        size_t new_grad(size_t);
        size_t grad_from_base(size_t, size_t, size_t);
        void delete_grad(size_t);

        const T* get_grad(size_t) const;

        void push_rvalue(cuda::unique_array<T>*, size_t);
        void push_lvalue(size_t);
    };
}

namespace grad::engine
{
    template<typename T>
    size_t tape<T>::push_grad(cuda::shared_array<T> *grad)
    {
        size_t index;
        if (_gaps.empty())
        {
            index = _next_index++;
            _gradients.push_back(grad);
        }
        else
        {
            index = _gaps.front();
            _gaps.pop_front();
            _gradients[index] = grad;
        }
        return index;
    }

    template<typename T>
    size_t tape<T>::new_grad(size_t size)
    {
        return push_grad(new cuda::shared_array<T>(size));
    }

    template<typename T>
    size_t tape<T>::grad_from_base(size_t base_index, size_t size, size_t offset)
    {
        return push_grad(new cuda::shared_array<T>(_gradients[base_index]->storage(), size, offset));
    }

    template<typename T>
    void tape<T>::delete_grad(size_t index)
    {
        _gaps.push_back(index);
        delete _gradients[index];
    }

    template<typename T>
    const T *tape<T>::get_grad(size_t index) const
    {
        return _gradients[index]->data();
    }

    template<typename T>
    void tape<T>::push_rvalue(cuda::unique_array<T> *mult, size_t index)
    {
        _multipliers.push_back(mult);
        _rvalue_indices.push_back(index);
    }

    template<typename T>
    void tape<T>::push_lvalue(size_t index)
    {
        _lvalue_indices.push_back(index);
        _last_op_indices.push_back(_rvalue_indices.size());
    }
}
