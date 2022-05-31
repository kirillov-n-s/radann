#pragma once
#include <vector>
#include <initializer_list>
#include <numeric>
#include <iostream>

namespace radann
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
