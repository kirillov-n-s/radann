#pragma once
#include <array>
#include <initializer_list>
#include <numeric>
#include <iostream>

namespace grad
{
    template <size_t N>
    class shape
    {
    public:
        using const_iterator = typename std::array<size_t, N>::const_iterator;
        using const_reverse_iterator = typename std::array<size_t, N>::const_reverse_iterator;
        static constexpr size_t rank = N;

    private:
        std::array<size_t, N> _data;
        size_t _length;

    public:
        template<typename InputIterator>
        shape(InputIterator, InputIterator);
        shape(const std::initializer_list<size_t>&);
        shape(const shape&) = default;
        shape& operator=(const shape&) = default;

        size_t operator[](size_t) const;

        size_t length() const;

        const_iterator begin() const;
        const_iterator end() const;

        const_reverse_iterator rbegin() const;
        const_reverse_iterator rend() const;

        template <size_t I>
        size_t offset(const shape<I>&);

        template <size_t I>
        shape<N - I> slice();

        template <size_t I>
        shape<N - I> flatten();
    };

    template <typename... Extents>
    shape<sizeof...(Extents)> make_shape(Extents...);

    template <size_t N>
    bool operator==(const shape<N>&, const shape<N>&);
    template <size_t N>
    bool operator!=(const shape<N>&, const shape<N>&);

    template <size_t N>
    std::ostream& operator<<(std::ostream&, const shape<N>&);
}

namespace grad
{
    template<size_t N>
    template<typename InputIterator>
    shape<N>::shape(InputIterator first, InputIterator last)
    {
        std::copy(first, last, _data.begin());
        _length = std::accumulate(_data.begin(), _data.end(), (size_t)1, std::multiplies<size_t>{});
    }

    template<size_t N>
    shape<N>::shape(const std::initializer_list<size_t> &data)
        : shape(data.begin(), data.end()) {}

    template<size_t N>
    size_t shape<N>::operator[](size_t i) const
    {
        return _data[i];
    }

    template<size_t N>
    size_t shape<N>::length() const
    {
        return _length;
    }

    template<size_t N>
    typename shape<N>::const_iterator shape<N>::begin() const
    {
        return _data.begin();
    }

    template<size_t N>
    typename shape<N>::const_iterator shape<N>::end() const
    {
        return _data.end();
    }

    template<size_t N>
    typename shape<N>::const_reverse_iterator shape<N>::rbegin() const
    {
        return _data.rbegin();
    }

    template<size_t N>
    typename shape<N>::const_reverse_iterator shape<N>::rend() const
    {
        return _data.rend();
    }

    template <size_t N>
    template <size_t I>
    size_t shape<N>::offset(const shape<I> &index)
    {
        static_assert(N >= I, "Index rank exceeds shape rank.");
        auto diff = N - I;
        auto acc = std::accumulate(_data.begin(), _data.begin() + diff, (size_t)1, std::multiplies<size_t>{});
        size_t res = 0;
        for (size_t i = 0; i < I; i++)
        {
            res += index[I - i - 1] * acc;
            acc *= _data[i];
        }
        return res;
    }

    template <size_t N>
    template <size_t I>
    shape<N - I> shape<N>::slice()
    {
        return { _data.begin(), _data.end() - I };
    }

    template <size_t N>
    template <size_t I>
    shape<N - I> shape<N>::flatten()
    {
        static_assert(N > I, "Flatten results in negative rank.");
        auto first = _data.begin();
        auto last = first + I + 1;
        std::array<size_t, N - I> data = { std::accumulate(first, last, (size_t)1, std::multiplies<size_t>{}) };
        std::copy(last, _data.end(), data.begin() + 1);
        return { data.begin(), data.end() };
    }

    template <typename... Extents>
    shape<sizeof...(Extents)> make_shape(Extents... extents)
    {
        return { (size_t)extents... };
    }

    template <size_t N>
    bool operator==(const shape<N>& lhs, const shape<N>& rhs)
    {
        return std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <size_t N>
    bool operator!=(const shape<N>& lhs, const shape<N>& rhs)
    {
        return !(lhs == rhs);
    }

    template <size_t N>
    std::ostream& operator<<(std::ostream &out, const shape<N> &shape)
    {
        if constexpr(N == 0)
            return out;

        out << '(';
        std::copy(shape.begin(), shape.end() - 1, std::ostream_iterator<size_t>(out, ", "));
        out << shape[N - 1];
        return out << ")\n";
    }
}
