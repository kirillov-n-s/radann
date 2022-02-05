#pragma once
#include <iomanip>
#include "shape.h"
#include "storage.h"
#include "kernel.h"
#include "xtmp/expr.h"

namespace grad
{
    template <typename T, size_t N>
    class array : public xtmp::expr<array<T, N>>
    {
    public:
        using value_type = T;
        static const size_t rank = N;

    private:
        mem::storage<T>* _storage;
        shape<N> _shape;
        size_t _offset = 0;

    public:
        array(const grad::shape<N>&, T = 0);
        template<typename InputIterator>
        array(const grad::shape<N>&, InputIterator, InputIterator);
        array(const grad::shape<N>&, const std::initializer_list<T>&);
        template<typename Expr>
        array(const xtmp::expr<Expr>&);

        array(mem::storage<T>*, const grad::shape<N>&, size_t);
        array(const array&);

        ~array();

        template<typename InputIterator>
        array& assign(InputIterator, InputIterator);

        array& operator=(const array&);
        array& operator=(const std::initializer_list<T>&);

        array& operator>>=(const array&);

        /*array& operator+=(const array&);
        array& operator-=(const array&);
        array& operator*=(const array&);
        array& operator/=(const array&);*/

        const grad::shape<N>& shape() const;
        size_t shape(size_t) const;

        template <typename... Indices>
        array<T, N - sizeof...(Indices)> operator()(Indices...);

        template<size_t M>
        array<T, M> reshape(const grad::shape<M>&);

        template<size_t I>
        array<T, N - I> flatten();

        const T* data() const;
        T* data();

        GRAD_DEVICE T operator[](size_t) const;
    };

    template<typename T, size_t N>
    std::ostream& operator<<(std::ostream&, const array<T, N>&);
}

namespace grad
{
    template<typename T, size_t N>
    array<T, N>::array(const grad::shape<N> &shape, T val)
        : _shape(shape),
        _storage(mem::make_storage(shape.length(), val))
    {}

    template<typename T, size_t N>
    template<typename InputIterator>
    array<T, N>::array(const grad::shape<N> &shape, InputIterator first, InputIterator last)
        : _shape(shape),
        _storage(mem::make_storage(shape.length(), T(0)))
    {
        if (std::distance(first, last) > _shape.length())
            throw std::invalid_argument("Iterator range exceeds array shape.");
        kernel::copy(first, last, data());
    }

    template<typename T, size_t N>
    array<T, N>::array(const grad::shape<N> &shape, const std::initializer_list<T> &data)
        : array(shape, data.begin(), data.end())
    {}

    template<typename T, size_t N>
    array<T, N>::array(mem::storage<T> *storage, const grad::shape<N> &shape, size_t offset)
        : _storage(storage), _shape(shape), _offset(offset)
    {
        _storage->add_ref();
    }

    template<typename T, size_t N>
    array<T, N>::array(const array &other)
        : array(other._storage, other._shape, other._offset)
    {}

    template<typename T, size_t N>
    template<typename Expr>
    array<T, N>::array(const xtmp::expr<Expr> &expr)
        : _shape(expr.self().shape()),
        _storage(mem::make_storage(expr.self().shape().length(), T(0)))
    {
        kernel::assign(data(), _shape.length(), expr.self());
    }

    template<typename T, size_t N>
    array<T, N>::~array()
    {
        _storage->remove_ref();
    }

    template<typename T, size_t N>
    template<typename InputIterator>
    array<T, N> &array<T, N>::assign(InputIterator first, InputIterator last)
    {
        if (std::distance(first, last) > _shape.length())
            throw std::invalid_argument("Iterator range exceeds array shape.");
        kernel::copy(first, last, data());
        return *this;
    }

    template<typename T, size_t N>
    array<T, N> &array<T, N>::operator=(const array &other)
    {
        if (this == &other)
            return *this;
        if (_shape != other._shape)
            throw std::invalid_argument("Array shape mismatch.");
        _storage->copy(other._storage->data(other._offset), _offset);
    }

    template<typename T, size_t N>
    array<T, N> &array<T, N>::operator=(const std::initializer_list<T> &data)
    {
        return assign(data.begin(), data.end());
    }

    template<typename T, size_t N>
    array<T, N> &array<T, N>::operator>>=(const array &other)
    {
        if (this == &other)
            return *this;
        _storage->remove_ref();
        _storage = other._storage;
        _storage->add_ref();
        _shape = other._shape;
        _offset = other._offset;
        return *this;
    }

    template<typename T, size_t N>
    const grad::shape<N> &array<T, N>::shape() const
    {
        return _shape;
    }

    template<typename T, size_t N>
    size_t array<T, N>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<typename T, size_t N>
    template<typename... Indices>
    array<T, N - sizeof...(Indices)> array<T, N>::operator()(Indices... indices)
    {
        auto index = make_shape(indices...);
        auto extents = _shape.template slice<index.rank>();
        auto offset = _shape.offset( index);
        return { _storage, extents, offset };
    }

    template<typename T, size_t N>
    template<size_t M>
    array<T, M> array<T, N>::reshape(const grad::shape<M> &shape)
    {
        if (_shape.length() != shape.length())
            throw std::invalid_argument("Array shape overall length mismatch.");
        return { _storage, shape, _offset };
    }

    template<typename T, size_t N>
    template<size_t I>
    array<T, N - I> array<T, N>::flatten()
    {
        return { _storage, _shape.template flatten<I>(), _offset };
    }

    template<typename T, size_t N>
    const T *array<T, N>::data() const
    {
        return _storage->data(_offset);
    }

    template<typename T, size_t N>
    T *array<T, N>::data()
    {
        return _storage->data(_offset);
    }

    template<typename T, size_t N>
    GRAD_DEVICE T array<T, N>::operator[](size_t i) const
    {
        return _storage->data(_offset)[i % _shape.length()];
    }

    template<typename T, size_t N>
    std::ostream &operator<<(std::ostream &out, const array<T, N> &array)
    {
        const auto& shape = array.shape();
        std::array<size_t, shape.rank> prods;
        std::partial_sum(shape.begin(), shape.end(), prods.begin(), std::multiplies<size_t>{});

        out << "0x" << array.data() << '\n';
        //out << std::scientific << std::setprecision(std::numeric_limits<T>::max_digits10) << std::right << std::showpos << std::setfill(' ');
        out << shape << '[' << array[0] << ", ";

        auto n = shape.length();
        for (size_t i = 1; i < n - 1; i++)
        {
            for (const auto& p : prods)
                out << (i % p == 0 ? "\n" : "");
            out << (i % prods[0] == 0 ? " " : "") << array[i] << ", ";
        }

        return out << array[n - 1] << "]\n\n";
    }
}
