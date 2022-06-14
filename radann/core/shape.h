#pragma once
#include <vector>
#include <initializer_list>
#include <numeric>
#include <iostream>

namespace radann::core
{
    class shape
    {
    public:
        using const_iterator = typename std::vector<size_t>::const_iterator;

    private:
        std::vector<size_t> _data;
        size_t _length;

    public:
        template<typename InputIterator>
        shape(InputIterator, InputIterator);
        shape(const std::initializer_list<size_t>&);
        shape(const shape&) = default;
        shape& operator=(const shape&) = default;

        size_t operator[](size_t) const;

        size_t length() const;
        size_t rank() const;

        const_iterator begin() const;
        const_iterator end() const;

        size_t offset(const shape&) const;
        size_t stride(size_t) const;

        shape cut(size_t) const;
        shape flatten(size_t) const;
    };

    template <typename... Extents>
    shape make_shape(Extents...);

    bool operator==(const shape&, const shape&);
    bool operator!=(const shape&, const shape&);

    std::ostream& operator<<(std::ostream&, const shape&);
}

namespace radann::core
{
    template<typename InputIterator>
    shape::shape(InputIterator first, InputIterator last)
        : _data(std::distance(first, last))
    {
        std::copy(first, last, _data.begin());
        _length = std::accumulate(_data.begin(), _data.end(), (size_t)1, std::multiplies<size_t>{});
    }

    shape::shape(const std::initializer_list<size_t> &data)
        : shape(data.begin(), data.end())
    {}

    size_t shape::operator[](size_t i) const
    {
        return i < rank() ? _data[i] : 1;
    }

    size_t shape::length() const
    {
        return _length;
    }

    size_t shape::rank() const
    {
        return _data.size();
    }

    typename shape::const_iterator shape::begin() const
    {
        return _data.begin();
    }

    typename shape::const_iterator shape::end() const
    {
        return _data.end();
    }

    size_t shape::offset(const shape &index) const
    {
        auto this_rank = rank();
        auto index_rank = index.rank();
        if (this_rank < index_rank)
            throw std::runtime_error("Index rank exceeds shape rank.");
        auto acc = std::accumulate(_data.begin(), _data.begin() + this_rank - index_rank,
                                   (size_t)1, std::multiplies<size_t>{});
        size_t res = 0;
        for (size_t i = 0; i < index_rank; i++)
        {
            res += index[index_rank - i - 1] * acc;
            acc *= _data[this_rank - index_rank + i];
        }
        return res;
    }

    size_t shape::stride(size_t i) const
    {
        if (rank() <= i)
            throw std::runtime_error("Stride dimension exceeds shape rank.");
        return std::accumulate(_data.begin(), _data.begin() + i, (size_t)1, std::multiplies<size_t>{});
    }

    shape shape::cut(size_t ndims) const
    {
        if (rank() < ndims)
            throw std::runtime_error("Cut results in negative rank.");
        return { _data.begin(), _data.end() - ndims };
    }

    shape shape::flatten(size_t ndims) const
    {
        auto rank = this->rank();
        if (rank <= ndims)
            throw std::runtime_error("Flatten results in negative rank.");

        auto first = _data.begin();
        auto last = first + ndims + 1;

        std::vector<size_t> data(rank - ndims);
        auto new_first = data.begin();
        auto new_last = data.end();

        *new_first = std::accumulate(first, last, (size_t)1, std::multiplies<size_t>{});
        std::copy(last, _data.end(), new_first + 1);

        return { new_first, new_last };
    }

    template <typename... Extents>
    shape make_shape(Extents... extents)
    {
        static_assert(std::conjunction_v<std::is_integral<Extents>...>, "Extents type must be integral.");
        return { (size_t)extents... };
    }

    bool operator==(const shape& lhs, const shape& rhs)
    {
        return lhs.length() == rhs.length()
            && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    bool operator!=(const shape& lhs, const shape& rhs)
    {
        return !(lhs == rhs);
    }

    std::ostream& operator<<(std::ostream &out, const shape &shape)
    {
        auto rank = shape.rank();
        if (rank == 0)
            return out;

        out << '(';
        std::copy(shape.begin(), shape.end() - 1, std::ostream_iterator<size_t>(out, ", "));
        out << shape[rank - 1];
        return out << ")\n";
    }
}
