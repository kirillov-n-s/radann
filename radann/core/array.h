#pragma once
#include <iomanip>
#include "shape.h"
#include "default.h"
#include "../cuda/shared_array.h"
#include "../cuda/assign.h"
#include "../diff/is_ad.h"

namespace radann::core
{
    template<typename T, typename Strategy>
    class array :
        public expr::base<array<T, Strategy>>,
        public cuda::shared_array<T>,
        public Strategy
    {
    public:
        using value_type = T;
        using strategy_type = Strategy;
        static constexpr bool is_expr = false;

    private:
        shape _shape;

    public:
        array(cuda::shared_storage<T>*, const core::shape&, size_t,
              const typename Strategy::index_type&, bool = true);
        //array(cuda::shared_storage<T>*, const core::shape&, size_t);
        array(const T*, const core::shape&, bool = autodiff);

        array(const core::shape&, bool = autodiff);
        template<typename InputIterator>
        array(const core::shape&, InputIterator, InputIterator, bool = autodiff);
        array(const core::shape&, const std::initializer_list<T>&, bool = autodiff);

        template<typename OtherStrategy>
        array(const array<T, OtherStrategy>&);

        template<typename Expr>
        array(const core::shape&, const expr::base<Expr>&, bool);
        template<typename Expr>
        array(const expr::base<Expr>&, bool);

        template<typename Expr>
        array(const core::shape&, const expr::base<Expr>&);
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

        //array_no_ad& operator>>=(const array_no_ad&);

        size_t rank() const;
        const shape& shape() const;
        size_t shape(size_t) const;

        array<T, Strategy> at(const core::shape&) const;
        template <typename... Indices>
        array<T, Strategy> operator()(Indices...) const;

        array<T, Strategy> reshape(const core::shape&) const;
        array<T, Strategy> flatten(size_t) const;
        array<T, Strategy> flatten() const;

        template<typename Op, typename Arg>
        friend auto eager(const Op&, const expr::base<Arg>&);

        template <typename Op, typename Lhs, typename Rhs>
        friend auto eager(const Op&, const expr::base<Lhs>&, const expr::base<Rhs>&);
    };

    template<typename T, typename Strategy>
    std::ostream& operator<<(std::ostream&, const array<T, Strategy>&);
}

namespace radann::core
{
    template<typename T, typename Strategy>
    array<T, Strategy>::array(cuda::shared_storage<T> *storage, const core::shape &shape, size_t offset,
                              const typename Strategy::index_type &base_index, bool derive)
        : cuda::shared_array<T>(storage, shape.length(), offset),
          Strategy(shape, offset, base_index, derive),
          _shape(shape)
    {}

    /*template<typename T, typename Strategy>
    array<T, Strategy>::array(cuda::shared_storage <T> *storage, const core::shape &shape, size_t offset)
        : cuda::shared_array<T>(storage, shape.length(), offset),
          Strategy(),
          _shape(shape)
    {}*/

    template<typename T, typename Strategy>
    array<T, Strategy>::array(const T *device_ptr, const core::shape &shape, bool ad)
        : cuda::shared_array<T>(device_ptr, shape.length()),
          Strategy(shape, ad),
          _shape(shape)
    {}

    template<typename T, typename Strategy>
    array<T, Strategy>::array(const core::shape &shape, bool ad)
        : cuda::shared_array<T>(shape.length()),
          Strategy(shape, ad),
          _shape(shape)
    {}

    template<typename T, typename Strategy>
    template<typename InputIterator>
    array<T, Strategy>::array(const core::shape &shape, InputIterator first, InputIterator last, bool ad)
        : cuda::shared_array<T>(shape.length()),
          Strategy(shape, ad),
          _shape(shape)
    {
        auto dist = std::distance(first, last);
        if (dist > this->_size)
            throw std::invalid_argument("Iterator range exceeds array_no_ad shape.");
        cuda::host_buffer<T> host { first, last };
        this->_storage->copy_from(host);
    }

    template<typename T, typename Strategy>
    array<T, Strategy>::array(const core::shape &shape, const std::initializer_list<T> &data, bool ad)
        : array(shape, data.begin(), data.end(), ad)
    {}

    template<typename T, typename Strategy>
    template<typename OtherStrategy>
    array<T, Strategy>::array(const array<T, OtherStrategy> &other)
        : array(other.storage(), other.shape(), other.offset(), other.grad_index(), false)
    {}

    template<typename T, typename Strategy>
    template<typename Expr>
    array<T, Strategy>::array(const core::shape &shape, const expr::base<Expr> &expr, bool ad)
        : cuda::shared_array<T>(shape.length()),
          Strategy(shape, ad),
          _shape(shape)
    {
        auto access = expr::get_access(expr.self());
        cuda::assign(this->data(), this->_size, access);
        if constexpr(Strategy::does_record)
            this->record_grad(access);
    }

    template<typename T, typename Strategy>
    template<typename Expr>
    array<T, Strategy>::array(const expr::base<Expr> &expr, bool ad)
        : array(expr.self().shape(), expr, ad)
    {}

    template<typename T, typename Strategy>
    template<typename Expr>
    array<T, Strategy>::array(const core::shape &shape, const expr::base<Expr> &expr)
        : array(expr.self().shape(), expr, diff::is_ad(expr.self()))
    {}

    template<typename T, typename Strategy>
    template<typename Expr>
    array<T, Strategy>::array(const expr::base<Expr> &expr)
        : array(expr.self().shape(), expr)
    {}

    template<typename T, typename Strategy>
    template<typename InputIterator>
    array<T, Strategy> &array<T, Strategy>::assign(InputIterator first, InputIterator last)
    {
        auto dist = std::distance(first, last);
        if (dist > this->_size)
            throw std::invalid_argument("Iterator range exceeds array_no_ad shape.");
        cuda::host_buffer<T> host { first, last };
        this->_storage->copy_from(host, this->_offset);
        return *this;
    }

    template<typename T, typename Strategy>
    array<T, Strategy> &array<T, Strategy>::operator=(const std::initializer_list<T> &data)
    {
        return assign(data.begin(), data.end());
    }

    template<typename T, typename Strategy>
    template<typename Expr>
    array<T, Strategy> &array<T, Strategy>::operator=(const expr::base<Expr> &expr)
    {
        auto access = expr::get_access(expr.self());
        if constexpr(Strategy::does_record)
            this->record_grad(access);
        cuda::assign(this->data(), this->_size, access);
        return *this;
    }

    template<typename T, typename Strategy>
    array<T, Strategy> &array<T, Strategy>::operator=(const array &other)
    {
        if (this == &other)
            return *this;
        return (*this = expr::get_access(other));
    }

    template<typename T, typename Strategy>
    template<typename Expr>
    array<T, Strategy> &array<T, Strategy>::operator+=(const expr::base<Expr> &expr)
    {
        return (*this = *this + expr);
    }

    template<typename T, typename Strategy>
    template<typename Expr>
    array<T, Strategy> &array<T, Strategy>::operator-=(const expr::base<Expr> &expr)
    {
        return (*this = *this - expr);
    }

    template<typename T, typename Strategy>
    template<typename Expr>
    array<T, Strategy> &array<T, Strategy>::operator*=(const expr::base<Expr> &expr)
    {
        return (*this = *this * expr);
    }

    template<typename T, typename Strategy>
    template<typename Expr>
    array<T, Strategy> &array<T, Strategy>::operator/=(const expr::base<Expr> &expr)
    {
        return (*this = *this / expr);
    }

    /*template<typename T, typename Strategy>
    array_no_ad<T, Strategy> &array_no_ad<T, Strategy>::operator>>=(const array_no_ad &other)
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
    }*/

    template<typename T, typename Strategy>
    size_t array<T, Strategy>::rank() const
    {
        return _shape.rank();
    }

    template<typename T, typename Strategy>
    const shape &array<T, Strategy>::shape() const
    {
        return _shape;
    }

    template<typename T, typename Strategy>
    size_t array<T, Strategy>::shape(size_t i) const
    {
        return _shape[i];
    }

    template<typename T, typename Strategy>
    array<T, Strategy> array<T, Strategy>::at(const radann::core::shape &index) const
    {
        auto extents = _shape.cut(index.rank());
        auto offset = _shape.offset(index);
        return array<T, Strategy> { this->_storage, extents, this->_offset + offset, this->grad_index() };
    }

    template<typename T, typename Strategy>
    template<typename... Indices>
    array<T, Strategy> array<T, Strategy>::operator()(Indices... indices) const
    {
        return at(make_shape(indices...));
    }

    template<typename T, typename Strategy>
    array<T, Strategy> array<T, Strategy>::reshape(const radann::core::shape &shape) const
    {
        if (this->_size != shape.length())
            throw std::invalid_argument("Array size mismatch.");
        return array<T, Strategy> { this->_storage, shape, this->_offset, this->grad_index() };
    }

    template<typename T, typename Strategy>
    array<T, Strategy> array<T, Strategy>::flatten(size_t ndims) const
    {
        return array<T, Strategy> { this->_storage, _shape.flatten(ndims), this->_offset, this->grad_index() };
    }

    template<typename T, typename Strategy>
    array<T, Strategy> array<T, Strategy>::flatten() const
    {
        return flatten(rank() - 1);
    }

    template<typename T, typename Strategy>
    std::ostream &operator<<(std::ostream &out, const array<T, Strategy> &array)
    {
        const auto host = array.host();
        const auto data = host.data();
        const auto storage = array.storage();

        /*out
            << std::scientific
            << std::setprecision(std::numeric_limits<T>::max_digits10)
            << std::right
            << std::showpos;*/

        out << "0x" << storage->data() << '\n'
            << "nrefs = " << storage->nrefs() << '\n';

        if (array.ad())
            out << "autodiff = true\n"
                << "gradient index = " << array.grad_index().value() << '\n';
        else
            out << "autodiff = false\n";

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
