#pragma once
#include <iomanip>
#include "default.h"
#include "shape.h"
#include "../cuda/assign.h"
#include "../engine/access.h"
#include "../engine/tape_context.h"
#include "sequence.h"

namespace radann
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
        array(const T*, const radann::shape<N>&);
        array(cuda::shared_storage<T>*, const radann::shape<N>&, size_t, size_t, bool = true);

        array(const radann::shape<N>&);
        template<typename InputIterator>
        array(const radann::shape<N>&, InputIterator, InputIterator);
        array(const radann::shape<N>&, const std::initializer_list<T>&);

        array(const array&);

        template<typename Expr>
        array(const radann::shape<N>&, const engine::expr<Expr>&);
        template<typename Expr>
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

        const radann::shape<N>& shape() const;
        size_t shape(size_t) const;

        template<size_t I>
        array<N - I, AD, T> at(const radann::shape<I>&) const;
        template <typename... Indices>
        array<N - sizeof...(Indices), AD, T> operator()(Indices...) const;

        template<size_t M>
        array<M, AD, T> reshape(const radann::shape<M>&) const;
        template<size_t I = N - 1>
        array<N - I, AD, T> flatten() const;

        size_t grad_index() const;
        array<N, false, T> get_grad() const;
        template<typename Expr>
        void set_grad(const engine::expr<Expr>&) const;
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
        array(const T*, const radann::shape<N>&);
        array(cuda::shared_storage<T>*, const radann::shape<N>&, size_t);

        array(const radann::shape<N>&);
        template<typename InputIterator>
        array(const radann::shape<N>&, InputIterator, InputIterator);
        array(const radann::shape<N>&, const std::initializer_list<T>&);

        array(const array&);

        template<typename Expr>
        array(const radann::shape<N>&, const engine::expr<Expr>&);
        template<typename Expr>
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

        const radann::shape<N>& shape() const;
        size_t shape(size_t) const;

        template<size_t I>
        array<N - I, false, T> at(const radann::shape<I>&) const;
        template <typename... Indices>
        array<N - sizeof...(Indices), false, T> operator()(Indices...) const;

        template<size_t M>
        array<M, false, T> reshape(const radann::shape<M>&) const;
        template<size_t I = N - 1>
        array<N - I, false, T> flatten() const;
    };

    template<bool AD = autodiff, typename T = real, size_t N>
    inline auto make_array(const radann::shape<N>&);

    template<bool AD = autodiff, typename InputIterator, size_t N>
    inline auto make_array(const radann::shape<N>&, InputIterator, InputIterator);
    template<bool AD = autodiff, size_t N, typename T>
    inline auto make_array(const radann::shape<N>&, const std::initializer_list<T>&);

    template <typename Expr, size_t N>
    inline auto make_array(const shape<N>&, const engine::expr<Expr>&);
    template <typename Expr>
    inline auto make_array(const engine::expr<Expr>&);

    template <bool AD, typename Expr, size_t N>
    inline auto make_array(const shape<N>&, const engine::expr<Expr>&);
    template <bool AD, typename Expr>
    inline auto make_array(const engine::expr<Expr>&);

    template<size_t N, bool AD, typename T>
    std::ostream& operator<<(std::ostream&, const array<N, AD, T>&);
}

namespace radann
{
    template<size_t N, bool AD, typename T>
    array<N, AD, T>::array(const T *device_ptr, const radann::shape<N> &shape)
        : cuda::shared_array<T>(device_ptr, shape.length()),
          _shape(shape),
          _grad_index(engine::get_tape<T>()->create_grad(shape.length()))
    {}

    template<size_t N, typename T>
    array<N, false, T>::array(const T *device_ptr, const radann::shape<N> &shape)
        : cuda::shared_array<T>(device_ptr, shape.length()),
          _shape(shape)
    {}

    template<size_t N, bool AD, typename T>
    array<N, AD, T>::array(cuda::shared_storage<T> *storage, const radann::shape<N> &shape,
                           size_t offset, size_t base_index, bool derive)
        : cuda::shared_array<T>(storage, shape.length(), offset),
          _shape(shape),
          _grad_index(derive
                      ? engine::get_tape<T>()->derive_grad(base_index, shape.length(), offset)
                      : base_index)
    {}

    template<size_t N, typename T>
    array<N, false, T>::array(cuda::shared_storage<T> *storage, const radann::shape<N> &shape, size_t offset)
        : cuda::shared_array<T>(storage, shape.length(), offset),
          _shape(shape)
    {}

    template<size_t N, bool AD, typename T>
    array<N, AD, T>::array(const radann::shape<N> &shape)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape),
          _grad_index(engine::get_tape<T>()->create_grad(shape.length()))
    {}

    template<size_t N, typename T>
    array<N, false, T>::array(const radann::shape<N> &shape)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape)
    {}

    template<size_t N, bool AD, typename T>
    template<typename InputIterator>
    array<N, AD, T>::array(const radann::shape<N> &shape, InputIterator first, InputIterator last)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape),
          _grad_index(engine::get_tape<T>()->create_grad(shape.length()))
    {
        auto dist = std::distance(first, last);
        if (dist > this->_size)
            throw std::invalid_argument("Iterator range exceeds array shape.");
        cuda::host_buffer<T> host { first, last };
        this->_storage->copy_from(host);
    }

    template<size_t N, typename T>
    template<typename InputIterator>
    array<N, false, T>::array(const radann::shape<N> &shape, InputIterator first, InputIterator last)
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
    array<N, AD, T>::array(const radann::shape<N> &shape, const std::initializer_list<T> &data)
        : array(shape, data.begin(), data.end())
    {}

    template<size_t N, typename T>
    array<N, false, T>::array(const radann::shape<N> &shape, const std::initializer_list<T> &data)
        : array(shape, data.begin(), data.end())
    {}

    template<size_t N, bool AD, typename T>
    array<N, AD, T>::array(const array &other)
        : array(other._storage, other._shape, other._offset, other._grad_index, false)
    {}

    template<size_t N, typename T>
    array<N, false, T>::array(const array &other)
        : array(other._storage, other._shape, other._offset)
    {}

    //todo: if expr is ad, compute get_grad of rvalue & push lvalue, do nothing otherwise
    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T>::array(const radann::shape<N> &shape, const engine::expr<Expr> &expr)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape),
          _grad_index(engine::get_tape<T>()->create_grad(shape.length()))
    {
        cuda::assign(this->data(), this->_size, engine::get_access(expr.self()));
        if constexpr(Expr::is_autodiff)
        {
            expr.self().propagate_grad(constant<T>(1));
            engine::get_tape<T>()->push_lvalue(_grad_index);
        }
    }

    template<size_t N, typename T>
    template<typename Expr>
    array<N, false, T>::array(const radann::shape<N> &shape, const engine::expr<Expr> &expr)
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
    template<typename Expr>
    array<N, false, T>::array(const engine::expr<Expr> &expr)
        : array(expr.self().shape(), expr)
    {}

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

    //todo: if expr is ad, compute grad of rvalue & push lvalue, probably clear get_grad data otherwise
    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T> &array<N, AD, T>::operator=(const engine::expr<Expr> &expr)
    {
        cuda::assign(this->data(), this->_size, engine::get_access(expr.self()));
        return *this;
    }

    template<size_t N, typename T>
    template<typename Expr>
    array<N, false, T> &array<N, false, T>::operator=(const engine::expr<Expr> &expr)
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

    template<size_t N, typename T>
    array<N, false, T> &array<N, false, T>::operator=(const array &other)
    {
        if (this == &other)
            return *this;
        return (*this = engine::get_access(other));
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T> &array<N, AD, T>::operator+=(const engine::expr<Expr> &expr)
    {
        return (*this = *this + expr);
    }

    template<size_t N, typename T>
    template<typename Expr>
    array<N, false, T> &array<N, false, T>::operator+=(const engine::expr<Expr> &expr)
    {
        return (*this = *this + expr);
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T> &array<N, AD, T>::operator-=(const engine::expr<Expr> &expr)
    {
        return (*this = *this - expr);
    }

    template<size_t N, typename T>
    template<typename Expr>
    array<N, false, T> &array<N, false, T>::operator-=(const engine::expr<Expr> &expr)
    {
        return (*this = *this - expr);
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T> &array<N, AD, T>::operator*=(const engine::expr<Expr> &expr)
    {
        return (*this = *this * expr);
    }

    template<size_t N, typename T>
    template<typename Expr>
    array<N, false, T> &array<N, false, T>::operator*=(const engine::expr<Expr> &expr)
    {
        return (*this = *this * expr);
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    array<N, AD, T> &array<N, AD, T>::operator/=(const engine::expr<Expr> &expr)
    {
        return (*this = *this / expr);
    }

    template<size_t N, typename T>
    template<typename Expr>
    array<N, false, T> &array<N, false, T>::operator/=(const engine::expr<Expr> &expr)
    {
        return (*this = *this / expr);
    }

    //todo: clear all previous gradient data, delete get_grad by index
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

    template<size_t N, typename T>
    array<N, false, T> &array<N, false, T>::operator>>=(const array &other)
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
    const radann::shape<N> &array<N, AD, T>::shape() const
    {
        return _shape;
    }

    template<size_t N, typename T>
    const radann::shape<N> &array<N, false, T>::shape() const
    {
        return _shape;
    }

    template<size_t N, bool AD, typename T>
    size_t array<N, AD, T>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<size_t N, typename T>
    size_t array<N, false, T>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<size_t N, bool AD, typename T>
    template<size_t I>
    array<N - I, AD, T> array<N, AD, T>::at(const radann::shape<I> &index) const
    {
        auto extents = _shape.template cut<I>();
        auto offset = _shape.offset(index);
        return array<N - I, AD, T> { this->_storage, extents, this->_offset + offset, _grad_index };
    }

    template<size_t N, typename T>
    template<size_t I>
    array<N - I, false, T> array<N, false, T>::at(const radann::shape<I> &index) const
    {
        auto extents = _shape.template cut<I>();
        auto offset = _shape.offset(index);
        return array<N - I, false, T> { this->_storage, extents, this->_offset + offset };
    }

    template<size_t N, bool AD, typename T>
    template<typename... Indices>
    array<N - sizeof...(Indices), AD, T> array<N, AD, T>::operator()(Indices... indices) const
    {
        return at(make_shape(indices...));
    }

    template<size_t N, typename T>
    template<typename... Indices>
    array<N - sizeof...(Indices), false, T> array<N, false, T>::operator()(Indices... indices) const
    {
        return at(make_shape(indices...));
    }

    template<size_t N, bool AD, typename T>
    template<size_t M>
    array<M, AD, T> array<N, AD, T>::reshape(const radann::shape<M> &shape) const
    {
        if (this->_size != shape.length())
            throw std::invalid_argument("Array size mismatch.");
        return array<M, AD, T> { this->_storage, shape, this->_offset, _grad_index };
    }

    template<size_t N, typename T>
    template<size_t M>
    array<M, false, T> array<N, false, T>::reshape(const radann::shape<M> &shape) const
    {
        if (this->_size != shape.length())
            throw std::invalid_argument("Array size mismatch.");
        return array<M, false, T> { this->_storage, shape, this->_offset };
    }

    template<size_t N, bool AD, typename T>
    template<size_t I>
    array<N - I, AD, T> array<N, AD, T>::flatten() const
    {
        return array<N - I, AD, T> { this->_storage, _shape.template flatten<I>(), this->_offset, _grad_index };
    }

    template<size_t N, typename T>
    template<size_t I>
    array<N - I, false, T> array<N, false, T>::flatten() const
    {
        return array<N - I, false, T> { this->_storage, _shape.template flatten<I>(), this->_offset };
    }

    template<size_t N, bool AD, typename T>
    size_t array<N, AD, T>::grad_index() const
    {
        return _grad_index;
    }

    template<size_t N, bool AD, typename T>
    array<N, false, T> array<N, AD, T>::get_grad() const
    {
        return array<N, false, T> { engine::get_tape<T>()->get_grad(_grad_index), _shape };
    }

    template<size_t N, bool AD, typename T>
    template<typename Expr>
    void array<N, AD, T>::set_grad(const engine::expr<Expr> &expr) const
    {
        engine::get_tape<T>()->set_grad(_grad_index, expr);
    }

    template<bool AD, typename T, size_t N>
    inline auto make_array(const shape<N>& shape)
    {
        return array<N, AD, T> { shape };
    }

    template<bool AD, typename InputIterator, size_t N>
    inline auto make_array(const radann::shape<N>& shape, InputIterator first, InputIterator last)
    {
        return array<N, AD, typename std::iterator_traits<InputIterator>::value_type> { shape, first, last };
    }

    template<bool AD, size_t N, typename T>
    inline auto make_array(const radann::shape<N>& shape, const std::initializer_list<T>& data)
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
        const auto data = host.data();
        const auto storage = array.storage();

        /*out
            << std::scientific
            << std::setprecision(std::numeric_limits<T>::max_digits10)
            << std::right
            << std::showpos;*/

        out << "0x" << storage->data() << '\n'
            << "nrefs = " << storage->nrefs() << '\n'
            << ad;

        if constexpr(N == 0)
            return out << '[' << data[0] << "]\n\n";

        const auto& shape = array.shape();
        std::array<size_t, N> prods;
        std::partial_sum(shape.begin(), shape.end(), prods.begin(), std::multiplies<size_t>{});

        out << shape << '[' << data[0];

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
