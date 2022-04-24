#pragma once
#include <iomanip>
#include "defaults.h"
#include "shape.h"
#include "../cuda/assign.h"
#include "../engine/access.h"
#include "../engine/tape_context.h"

namespace grad
{
    template<size_t N, bool AD = autodiff, typename T = real>
    class array :
            public engine::expr<array<N, AD, T>>,
            public cuda::shared_array<T>
    {
    public:
        using value_type = T;
        static constexpr size_t rank = N;
        static constexpr bool is_expr = false;
        static constexpr bool is_autodiff = AD;

    private:
        shape<N> _shape;
        size_t _grad_index;

    public:
        array(cuda::shared_storage<T>*, const grad::shape<N>&, size_t);
        array(cuda::shared_storage<T>*, const grad::shape<N>&, size_t, size_t);

        array(const grad::shape<N>&);
        template<typename InputIterator>
        array(const grad::shape<N>&, InputIterator, InputIterator);
        array(const grad::shape<N>&, const std::initializer_list<T>&);

        array(const array&);
        array(const array<N, false, T>&);

        template<typename Expr>
        array(const grad::shape<N>&, const engine::expr<Expr>&);
        template<typename Expr>
        array(const engine::expr<Expr>&);

        ~array();

        template<typename InputIterator>
        array& assign(InputIterator, InputIterator);
        array& operator=(const std::initializer_list<T>&);

        template<typename Expr>
        array& operator=(const engine::expr<Expr>&);
        array& operator=(const array&);
        template<typename Expr>
        array& operator+=(const engine::expr<Expr>&);
        template<typename Expr>
        array& operator-=(const engine::expr<Expr>&);
        template<typename Expr>
        array& operator*=(const engine::expr<Expr>&);
        template<typename Expr>
        array& operator/=(const engine::expr<Expr>&);

        array& operator>>=(const array&);

        const grad::shape<N>& shape() const;
        size_t shape(size_t) const;

        template<size_t I>
        array<N - I, AD, T> at(const grad::shape<I>&) const;
        template <typename... Indices>
        array<N - sizeof...(Indices), AD, T> operator()(Indices...) const;

        template<size_t M>
        array<M, AD, T> reshape(const grad::shape<M>&) const;
        template<size_t I = N - 1>
        array<N - I, AD, T> flatten() const;

        array<N, false, T> grad() const;
    };

    template<size_t N, typename T>
    class array<N, false, T> :
            public engine::expr<array<N, false, T>>,
            public cuda::shared_array<T>
    {
    public:
        using value_type = T;
        static constexpr size_t rank = N;
        static constexpr bool is_expr = false;
        static constexpr bool is_autodiff = false;

    private:
        shape<N> _shape;

    public:
        array(cuda::shared_storage<T>*, const grad::shape<N>&, size_t);

        array(const grad::shape<N>&);
        template<typename InputIterator>
        array(const grad::shape<N>&, InputIterator, InputIterator);
        array(const grad::shape<N>&, const std::initializer_list<T>&);

        array(const array&);

        template<typename Expr, typename std::enable_if_t<!Expr::is_autodiff>* = nullptr>
        array(const grad::shape<N>&, const engine::expr<Expr>&);
        template<typename Expr, typename std::enable_if_t<!Expr::is_autodiff>* = nullptr>
        array(const engine::expr<Expr>&);

        ~array() = default;

        template<typename InputIterator>
        array& assign(InputIterator, InputIterator);
        array& operator=(const std::initializer_list<T>&);

        template<typename Expr>
        array& operator=(const engine::expr<Expr>&);
        array& operator=(const array&);
        template<typename Expr>
        array& operator+=(const engine::expr<Expr>&);
        template<typename Expr>
        array& operator-=(const engine::expr<Expr>&);
        template<typename Expr>
        array& operator*=(const engine::expr<Expr>&);
        template<typename Expr>
        array& operator/=(const engine::expr<Expr>&);

        array& operator>>=(const array&);

        const grad::shape<N>& shape() const;
        size_t shape(size_t) const;

        template<size_t I>
        array<N - I, false, T> at(const grad::shape<I>&) const;
        template <typename... Indices>
        array<N - sizeof...(Indices), false, T> operator()(Indices...) const;

        template<size_t M>
        array<M, false, T> reshape(const grad::shape<M>&) const;
        template<size_t I = N - 1>
        array<N - I, false, T> flatten() const;
    };

    template<bool AD = autodiff, typename T = real, size_t N>
    inline auto make_array(const grad::shape<N>&);

    template<bool AD = autodiff, typename InputIterator, size_t N>
    inline auto make_array(const grad::shape<N>&, InputIterator, InputIterator);
    template<bool AD = autodiff, size_t N, typename T>
    inline auto make_array(const grad::shape<N>&, const std::initializer_list<T>&);

    template <typename Expr, size_t N>
    inline auto make_array(const shape<N>&, const engine::expr<Expr>&);
    template <typename Expr>
    inline auto make_array(const engine::expr<Expr>&);

    template <bool AD = autodiff, typename Expr, size_t N>
    inline auto make_array(const shape<N>&, const engine::expr<Expr>&);
    template <bool AD = autodiff, typename Expr>
    inline auto make_array(const engine::expr<Expr>&);

    template<size_t N, bool AD, typename T>
    std::ostream& operator<<(std::ostream&, const array<N, AD, T>&);
}

namespace grad
{
    template<size_t N, bool AD, typename T>
    array<N, AD, T>::array(cuda::shared_storage<T> *storage, const grad::shape<N> &shape, size_t offset)
        : cuda::shared_array<T>(storage, shape.length(), offset),
          _shape(shape),
          _grad_index(engine::get_tape<T>()->new_grad(shape.length()))
    {}

    template<size_t N, bool AD, typename T>
    array<N, AD, T>::array(cuda::shared_storage<T> *storage, const grad::shape<N> &shape, size_t offset, size_t base_index)
        : cuda::shared_array<T>(storage, shape.length(), offset),
          _shape(shape),
          _grad_index(engine::get_tape<T>()->grad_from_base(base_index, shape.length(), offset))
    {}

    template<size_t N, typename T>
    array<N, false, T>::array(cuda::shared_storage<T> *storage, const grad::shape<N> &shape, size_t offset)
        : cuda::shared_array<T>(storage, shape.length(), offset),
          _shape(shape)
    {}

    template<size_t N, bool AD, typename T>
    array<N, AD, T>::array(const grad::shape<N> &shape)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape),
          _grad_index(engine::get_tape<T>()->new_grad(shape.length()))
    {}

    template<size_t N, typename T>
    array<N, false, T>::array(const grad::shape<N> &shape)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape)
    {}

    template<size_t N, bool AD, typename T>
    template<typename InputIterator>
    array<N, AD, T>::array(const grad::shape<N> &shape, InputIterator first, InputIterator last)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape),
          _grad_index(engine::get_tape<T>()->new_grad(shape.length()))
    {
        auto dist = std::distance(first, last);
        if (dist > this->_size)
            throw std::invalid_argument("Iterator range exceeds array shape.");
        cuda::host_buffer<T> host { first, last };
        this->_storage->copy_from(host);
    }

    template<size_t N, typename T>
    template<typename InputIterator>
    array<N, false, T>::array(const grad::shape<N> &shape, InputIterator first, InputIterator last)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape)
    {
        auto dist = std::distance(first, last);
        if (dist > this->_size)
            throw std::invalid_argument("Iterator range exceeds array shape.");
        cuda::host_buffer<T> host { first, last };
        this->_storage->copy_from(host);
    }

    template<size_t N, bool AD, typename T>
    array<N, AD, T>::array(const grad::shape<N> &shape, const std::initializer_list<T> &data)
        : array(shape, data.begin(), data.end())
    {}

    template<size_t N, typename T>
    array<N, false, T>::array(const grad::shape<N> &shape, const std::initializer_list<T> &data)
        : array(shape, data.begin(), data.end())
    {}

    template<size_t N, bool AD, typename T>
    array<N, AD, T>::array(const array &other)
        : array(other._storage, other._shape, other._offset, other._grad_index)
    {}

    //todo: decide if this should be allowed or not
    template<size_t N, bool AD, typename T>
    array<N, AD, T>::array(const array<N, false, T> &other)
        : array(other._storage, other._shape, other._offset)
    {}

    template<size_t N, typename T>
    array<N, false, T>::array(const array &other)
        : array(other._storage, other._shape, other._offset)
    {}

    //todo: if expr is ad, compute grad of rvalue & push lvalue, do nothing otherwise
    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T>::array(const grad::shape<N> &shape, const engine::expr<Expr> &expr)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape),
          _grad_index(engine::get_tape<T>()->new_grad(shape.length()))
    {
        cuda::assign(this->data(), this->_size, engine::get_access(expr.self()));
    }

    template<size_t N, typename T>
    template<typename Expr, std::enable_if_t<!Expr::is_autodiff>*>
    array<N, false, T>::array(const grad::shape<N> &shape, const engine::expr<Expr> &expr)
            : cuda::shared_array<T>(shape.length()),
              _shape(shape)
    {
        cuda::assign(this->data(), this->_size, engine::get_access(expr.self()));
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T>::array(const engine::expr<Expr> &expr)
        : array(expr.self().shape(), expr)
    {}

    template<size_t N, typename T>
    template<typename Expr, std::enable_if_t<!Expr::is_autodiff>*>
    array<N, false, T>::array(const engine::expr<Expr> &expr)
        : array(expr.self().shape(), expr)
    {}

    template<size_t N, bool AD, typename T>
    array<N, AD, T>::~array()
    {
        engine::get_tape<T>()->delete_grad(_grad_index);
    }

    //todo: should probably clear all gradient data
    template<size_t N, bool AD, typename T>
    template<typename InputIterator>
    array<N, AD, T> &array<N, AD, T>::assign(InputIterator first, InputIterator last)
    {
        auto dist = std::distance(first, last);
        if (dist > this->_size)
            throw std::invalid_argument("Iterator range exceeds array shape.");
        cuda::host_buffer<T> host { first, last };
        this->_storage->copy_from(host, this->_offset);
        return *this;
    }

    template<size_t N, typename T>
    template<typename InputIterator>
    array<N, false, T> &array<N, false, T>::assign(InputIterator first, InputIterator last)
    {
        auto dist = std::distance(first, last);
        if (dist > this->_size)
            throw std::invalid_argument("Iterator range exceeds array shape.");
        cuda::host_buffer<T> host { first, last };
        this->_storage->copy_from(host, this->_offset);
        return *this;
    }

    template<size_t N, bool AD, typename T>
    array<N, AD, T> &array<N, AD, T>::operator=(const std::initializer_list<T> &data)
    {
        return assign(data.begin(), data.end());
    }

    template<size_t N, typename T>
    array<N, false, T> &array<N, false, T>::operator=(const std::initializer_list<T> &data)
    {
        return assign(data.begin(), data.end());
    }

    //todo: if expr is ad, compute grad of rvalue & push lvalue, probably clear grad data otherwise
    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T> &array<N, AD, T>::operator=(const engine::expr<Expr> &expr)
    {
        cuda::assign(this->data(), this->_size, engine::get_access(expr.self()));
        return *this;
    }

    template<size_t N, bool AD, typename T>
    array<N, AD, T> &array<N, AD, T>::operator=(const array &other)
    {
        if (this == &other)
            return *this;
        return (*this = engine::get_access(other));
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T> &array<N, AD, T>::operator+=(const engine::expr<Expr> &expr)
    {
        cuda::assign(this->data(), this->_size, *this + expr);
        return *this;
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T> &array<N, AD, T>::operator-=(const engine::expr<Expr> &expr)
    {
        cuda::assign(this->data(), this->_size, *this - expr);
        return *this;
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T> &array<N, AD, T>::operator*=(const engine::expr<Expr> &expr)
    {
        cuda::assign(this->data(), this->_size, *this * expr);
        return *this;
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T> &array<N, AD, T>::operator/=(const engine::expr<Expr> &expr)
    {
        cuda::assign(this->data(), this->_size, *this / expr);
        return *this;
    }

    template<size_t N, bool AD, typename T>
    array<N, AD, T> &array<N, AD, T>::operator>>=(const array &other)
    {
        if (this == &other)
            return *this;
        this->_storage->remove_ref();
        this->_storage = other._storage;
        this->_storage->add_ref();
        _shape = other._shape;
        this->_offset = other._offset;
        return *this;
    }

    template<size_t N, bool AD, typename T>
    const grad::shape<N> &array<N, AD, T>::shape() const
    {
        return _shape;
    }

    template<size_t N, typename T>
    const grad::shape<N> &array<N, false, T>::shape() const
    {
        return _shape;
    }

    template<size_t N, bool AD, typename T>
    size_t array<N, AD, T>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<size_t N, bool AD, typename T>
    template<size_t I>
    array<N - I, AD, T> array<N, AD, T>::at(const grad::shape<I> &index) const
    {
        auto extents = _shape.template cut<I>();
        auto offset = _shape.offset(index);
        return array<N - I, AD, T> { this->_storage, extents, this->_offset + offset, _grad_index };
    }

    template<size_t N, bool AD, typename T>
    template<typename... Indices>
    array<N - sizeof...(Indices), AD, T> array<N, AD, T>::operator()(Indices... indices) const
    {
        return at(make_shape(indices...));
    }

    template<size_t N, bool AD, typename T>
    template<size_t M>
    array<M, AD, T> array<N, AD, T>::reshape(const grad::shape<M> &shape) const
    {
        if (this->_size != shape.length())
            throw std::invalid_argument("Array size mismatch.");
        return array<M, AD, T> { this->_storage, shape, this->_offset, _grad_index };
    }

    template<size_t N, bool AD, typename T>
    template<size_t I>
    array<N - I, AD, T> array<N, AD, T>::flatten() const
    {
        return array<N - I, AD, T> { this->_storage, _shape.template flatten<I>(), this->_offset, _grad_index };
    }

    template<size_t N, bool AD, typename T>
    array<N, false, T> array<N, AD, T>::grad() const
    {
        return array<N, false, T>
            { cuda::make_storage(engine::get_tape<T>()->get_grad(_grad_index), this->_size), _shape, this->_offset };
    }

    template<bool AD, typename T, size_t N>
    inline auto make_array(const shape<N>& shape)
    {
        return array<N, AD, T> { shape };
    }

    template<bool AD, typename InputIterator, size_t N>
    inline auto make_array(const grad::shape<N>& shape, InputIterator first, InputIterator last)
    {
        return array<N, AD, typename std::iterator_traits<InputIterator>::value_type> { shape, first, last };
    }

    template<bool AD, size_t N, typename T>
    inline auto make_array(const grad::shape<N>& shape, const std::initializer_list<T>& data)
    {
        return array<N, AD, T> { shape, data };
    }

    template<typename Expr, size_t N>
    inline auto make_array(const shape<N>& shape, const engine::expr<Expr>& expr)
    {
        return array<N, Expr::is_autodiff, typename Expr::value_type> { shape, expr };
    }

    template<typename Expr>
    inline auto make_array(const engine::expr<Expr>& expr)
    {
        return array<Expr::rank, Expr::is_autodiff, typename Expr::value_type> { expr.self().shape(), expr };
    }

    template<bool AD, typename Expr, size_t N>
    inline auto make_array(const shape<N>& shape, const engine::expr<Expr>& expr)
    {
        return array<N, AD, typename Expr::value_type> { shape, expr };
    }

    template<bool AD, typename Expr>
    inline auto make_array(const engine::expr<Expr>& expr)
    {
        return array<Expr::rank, AD, typename Expr::value_type> { expr.self().shape(), expr };
    }

    template<size_t N, bool AD, typename T>
    std::ostream &operator<<(std::ostream &out, const array<N, AD, T> &array)
    {
        constexpr auto ad = AD ? "AD = true\n" : "AD = false\n";

        const auto host = array.host();
        auto data = host.data();
        /*out
            << std::scientific
            << std::setprecision(std::numeric_limits<T>::max_digits10)
            << std::right
            << std::showpos;*/
        if constexpr(N == 0)
            return out << ad << '[' << data[0] << "]\n\n";

        const auto& shape = array.shape();
        std::array<size_t, shape.rank> prods;
        std::partial_sum(shape.begin(), shape.end(), prods.begin(), std::multiplies<size_t>{});

        out << ad << shape << '[' << data[0];

        auto n = shape.length();
        if (n > 1)
            out << ", ";
        else
            return out << "]\n\n";

        for (size_t i = 1; i < n - 1; i++)
        {
            for (const auto& p : prods)
                out << (i % p == 0 ? "\n" : "");
            out << (i % prods[0] == 0 ? " " : "") << data[i] << ", ";
        }

        return out << data[n - 1] << "]\n\n";
    }
}