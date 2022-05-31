#pragma once
#include <iomanip>
#include <optional>
#include "default.h"
#include "shape.h"
#include "../cuda/shared_array.h"
#include "../cuda/assign.h"
#include "../expr/access.h"
#include "../diff/tape_context.h"
#include "sequence.h"

namespace radann
{
    template<typename T = real>
    class array :
        public expr::base<array<T>>,
        public cuda::shared_array<T>
    {
    public:
        using value_type = T;
        static constexpr bool is_expr = false;

    private:
        shape _shape;
        std::optional<size_t> _grad_index = std::nullopt;

        array(cuda::shared_storage<T>*, const radann::shape&, size_t, const std::optional<size_t>&, bool = true);

    public:
        array(const radann::shape&, bool = autodiff);
        template<typename InputIterator>
        array(const radann::shape&, InputIterator, InputIterator, bool = autodiff);
        array(const radann::shape&, const std::initializer_list<T>&, bool = autodiff);

        array(const array&);

        template<typename Expr>
        array(const radann::shape&, const expr::base<Expr>&, bool);
        template<typename Expr>
        array(const expr::base<Expr>&, bool);

        template<typename Expr>
        array(const radann::shape&, const expr::base<Expr>&);
        template<typename Expr>
        array(const expr::base<Expr>&);

        ~array() = default;

        template<typename InputIterator>
        array& assign(InputIterator, InputIterator);
        array& operator=(const std::initializer_list<T>&);

        template<typename Expr>
        array& operator=(const expr::base<Expr>&);
        array& operator=(const array&);
        template<typename Expr>
        array& operator+=(const expr::base<Expr>&);
        template<typename Expr>
        array& operator-=(const expr::base<Expr>&);
        template<typename Expr>
        array& operator*=(const expr::base<Expr>&);
        template<typename Expr>
        array& operator/=(const expr::base<Expr>&);

        array& operator>>=(const array&);

        size_t rank() const;
        const radann::shape& shape() const;
        size_t shape(size_t) const;

        array<T> at(const radann::shape&) const;
        template <typename... Indices>
        array<T> operator()(Indices...) const;

        array<T> reshape(const radann::shape&) const;
        array<T> flatten(size_t) const;
        array<T> flatten() const;

        bool ad() const;
        const std::optional<size_t>& grad_index() const;
        array<T> get_grad() const;
        template<typename Expr>
        void set_grad(const expr::base<Expr>&) const;
    };

    template<typename T = real>
    inline auto make_array(const radann::shape&, bool = autodiff);

    template<typename InputIterator>
    inline auto make_array(const radann::shape&, InputIterator, InputIterator, bool = autodiff);
    template<typename T>
    inline auto make_array(const radann::shape&, const std::initializer_list<T>&, bool = autodiff);

    template <typename Expr>
    inline auto make_array(const shape&, const expr::base<Expr>&);
    template <typename Expr>
    inline auto make_array(const expr::base<Expr>&);

    template <typename Expr>
    inline auto make_array(const shape&, const expr::base<Expr>&, bool);
    template <typename Expr>
    inline auto make_array(const expr::base<Expr>&, bool);

    template<typename T>
    std::ostream& operator<<(std::ostream&, const array<T>&);
}

namespace radann
{
    template<typename T>
    array<T>::array(cuda::shared_storage<T> *storage, const radann::shape &shape, size_t offset,
                    const std::optional<size_t>& base_index, bool derive)
        : cuda::shared_array<T>(storage, shape.length(), offset),
          _shape(shape),
          _grad_index(!base_index.has_value()
                      ? std::nullopt
                      : (derive
                         ? diff::get_tape<T>()->derive_grad(base_index.value(), shape, offset)
                         : base_index))
    {}

    template<typename T>
    array<T>::array(const radann::shape &shape, bool ad)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape),
          _grad_index(ad
                      ? diff::get_tape<T>()->create_grad(shape)
                      : std::nullopt)
    {}

    template<typename T>
    template<typename InputIterator>
    array<T>::array(const radann::shape &shape, InputIterator first, InputIterator last, bool ad)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape),
          _grad_index(ad
                      ? diff::get_tape<T>()->create_grad(shape)
                      : std::nullopt)
    {
        auto dist = std::distance(first, last);
        if (dist > this->_size)
            throw std::invalid_argument("Iterator range exceeds array shape.");
        cuda::host_buffer<T> host { first, last };
        this->_storage->copy_from(host);
    }

    template<typename T>
    array<T>::array(const radann::shape &shape, const std::initializer_list<T> &data, bool ad)
        : array(shape, data.begin(), data.end(), ad)
    {}

    template<typename T>
    array<T>::array(const array &other)
        : array(other._storage, other._shape, other._offset, other._grad_index, false)
    {}

    template<typename T>
    template<typename Expr>
    array<T>::array(const radann::shape &shape, const expr::base<Expr> &expr, bool ad)
        : cuda::shared_array<T>(shape.length()),
          _shape(shape)
    {
        cuda::assign(this->data(), this->_size, expr::get_access(expr.self()));
        if (ad)
        {
            _grad_index = diff::get_tape<T>()->create_grad(shape);
            expr.self().propagate_grad(constant<T>(1));
            diff::get_tape<T>()->push_lvalue(_grad_index);
        }
    }

    template<typename T>
    template<typename Expr>
    array<T>::array(const expr::base<Expr> &expr, bool ad)
        : array(expr.self().shape(), expr, ad)
    {}

    template<typename T>
    template<typename Expr>
    array<T>::array(const radann::shape &shape, const expr::base<Expr> &expr)
        : array(expr.self().shape(), expr, expr.self().ad())
    {}

    template<typename T>
    template<typename Expr>
    array<T>::array(const expr::base<Expr> &expr)
        : array(expr.self().shape(), expr)
    {}

    template<typename T>
    template<typename InputIterator>
    array<T> &array<T>::assign(InputIterator first, InputIterator last)
    {
        auto dist = std::distance(first, last);
        if (dist > this->_size)
            throw std::invalid_argument("Iterator range exceeds array shape.");
        cuda::host_buffer<T> host { first, last };
        this->_storage->copy_from(host, this->_offset);
        return *this;
    }

    template<typename T>
    array<T> &array<T>::operator=(const std::initializer_list<T> &data)
    {
        return assign(data.begin(), data.end());
    }

    template<typename T>
    template<typename Expr>
    array<T> &array<T>::operator=(const expr::base<Expr> &expr)
    {
        auto expr_self = expr.self();
        cuda::assign(this->data(), this->_size, expr::get_access(expr_self));
        if (expr_self.ad() && ad())
        {
            expr_self.propagate_grad(constant<T>(1));
            diff::get_tape<T>()->push_lvalue(_grad_index.value());
        }
        return *this;
    }

    template<typename T>
    array<T> &array<T>::operator=(const array &other)
    {
        if (this == &other)
            return *this;
        return (*this = expr::get_access(other));
    }

    template<typename T>
    template<typename Expr>
    array<T> &array<T>::operator+=(const expr::base<Expr> &expr)
    {
        return (*this = *this + expr);
    }

    template<typename T>
    template<typename Expr>
    array<T> &array<T>::operator-=(const expr::base<Expr> &expr)
    {
        return (*this = *this - expr);
    }

    template<typename T>
    template<typename Expr>
    array<T> &array<T>::operator*=(const expr::base<Expr> &expr)
    {
        return (*this = *this * expr);
    }

    template<typename T>
    template<typename Expr>
    array<T> &array<T>::operator/=(const expr::base<Expr> &expr)
    {
        return (*this = *this / expr);
    }

    template<typename T>
    array<T> &array<T>::operator>>=(const array &other)
    {
        if (this == &other)
            return *this;
        this->_storage->remove_ref();
        this->_storage = other._storage;
        this->_storage->add_ref();
        this->_offset = other._offset;
        _shape = other._shape;
        _grad_index = other._grad_index;
        return *this;
    }

    template<typename T>
    size_t array<T>::rank() const
    {
        return _shape.rank();
    }

    template<typename T>
    const radann::shape &array<T>::shape() const
    {
        return _shape;
    }

    template<typename T>
    size_t array<T>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<typename T>
    array<T> array<T>::at(const radann::shape &index) const
    {
        auto extents = _shape.cut(index.rank());
        auto offset = _shape.offset(index);
        return array<T> { this->_storage, extents, this->_offset + offset, _grad_index };
    }

    template<typename T>
    template<typename... Indices>
    array<T> array<T>::operator()(Indices... indices) const
    {
        return at(make_shape(indices...));
    }

    template<typename T>
    array<T> array<T>::reshape(const radann::shape &shape) const
    {
        if (this->_size != shape.length())
            throw std::invalid_argument("Array size mismatch.");
        return array<T> { this->_storage, shape, this->_offset, _grad_index };
    }

    template<typename T>
    array<T> array<T>::flatten(size_t ndims) const
    {
        return array<T> { this->_storage, _shape.flatten(ndims), this->_offset, _grad_index };
    }

    template<typename T>
    array<T> array<T>::flatten() const
    {
        return flatten(rank() - 1);
    }

    template<typename T>
    bool array<T>::ad() const
    {
        return _grad_index.has_value();
    }

    template<typename T>
    const std::optional<size_t> &array<T>::grad_index() const
    {
        return _grad_index;
    }

    template<typename T>
    array<T> array<T>::get_grad() const
    {
        if (!ad())
            throw std::runtime_error("Array is not differentiated.");
        return diff::get_tape<T>()->get_grad(_grad_index);
    }

    template<typename T>
    template<typename Expr>
    void array<T>::set_grad(const expr::base<Expr> &expr) const
    {
        if (!ad())
            throw std::runtime_error("Array is not differentiated.");
        diff::get_tape<T>()->set_grad(_grad_index, expr);
    }

    template<typename T>
    inline auto make_array(const shape& shape, bool ad)
    {
        return array<T> { shape, ad };
    }

    template<typename InputIterator>
    inline auto make_array(const radann::shape& shape, InputIterator first, InputIterator last, bool ad)
    {
        return array<typename std::iterator_traits<InputIterator>::value_type> { shape, first, last, ad };
    }

    template<typename T>
    inline auto make_array(const radann::shape& shape, const std::initializer_list<T>& data, bool ad)
    {
        return array<T> { shape, data, ad };
    }

    template<typename Expr>
    inline auto make_array(const shape& shape, const expr::base<Expr>& expr)
    {
        return array<typename Expr::value_type> { shape, expr };
    }

    template<typename Expr>
    inline auto make_array(const expr::base<Expr>& expr)
    {
        return array<typename Expr::value_type> { expr };
    }

    template<typename Expr>
    inline auto make_array(const shape& shape, const expr::base<Expr>& expr, bool ad)
    {
        return array<typename Expr::value_type> { shape, expr, ad };
    }

    template<typename Expr>
    inline auto make_array(const expr::base<Expr>& expr, bool ad)
    {
        return array<typename Expr::value_type> { expr, ad };
    }

    template<typename T>
    std::ostream &operator<<(std::ostream &out, const array<T> &array)
    {
        auto ad = array.ad() ? "AD = true\n" : "AD = false\n";

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

        auto rank = array.rank();
        if (rank == 0)
            return out << '[' << data[0] << "]\n\n";

        const auto& shape = array.shape();
        std::vector<size_t> prods(rank);
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
