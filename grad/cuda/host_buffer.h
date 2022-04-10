#pragma once
#include <memory>
#include "cuda_runtime.h"

namespace grad::cuda
{
    template <typename T>
    class host_buffer
    {
    public:
        using value_type = T;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    private:
        std::unique_ptr<T[]> _data;
        size_t _size;
        size_t _nbytes;

    public:
        template<typename InputIterator>
        host_buffer(InputIterator, InputIterator);
        host_buffer(const T*, size_t);
        host_buffer(const host_buffer&) = default;
        ~host_buffer() = default;

        reference operator[](size_type);
        const_reference operator[](size_type) const;

        T* data();
        const T* data() const;

        iterator begin();
        const_iterator begin() const;

        iterator end();
        const_iterator end() const;

        reverse_iterator rbegin();
        const_reverse_iterator rbegin() const;

        reverse_iterator rend();
        const_reverse_iterator rend() const;

        size_type size() const;
        size_type nbytes() const;
    };
}

namespace grad::cuda
{
    template<typename T>
    template<typename InputIterator>
    host_buffer<T>::host_buffer(InputIterator first, InputIterator last)
    {
        auto dist = std::distance(first, last);
        _data = std::make_unique<T[]>(dist);
        _size = dist;
        _nbytes = dist * sizeof(T);
        std::copy(first, last, _data.get());
    }

    template<typename T>
    host_buffer<T>::host_buffer(const T *device_ptr, size_t size)
        : _data(std::make_unique<T[]>(size)), _size(size), _nbytes(size * sizeof(T))
    {
        auto status = cudaMemcpy(_data.get(), device_ptr, _nbytes,
                                 cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::bad_alloc();
    }

    template<typename T>
    typename host_buffer<T>::reference host_buffer<T>::operator[](host_buffer::size_type i)
    {
        return _data[i];
    }

    template<typename T>
    typename host_buffer<T>::const_reference host_buffer<T>::operator[](host_buffer::size_type i) const
    {
        return _data[i];
    }

    template<typename T>
    T *host_buffer<T>::data()
    {
        return _data.get();
    }

    template<typename T>
    const T *host_buffer<T>::data() const
    {
        return _data.get();
    }

    template<typename T>
    typename host_buffer<T>::iterator host_buffer<T>::begin()
    {
        return _data.get();
    }

    template<typename T>
    typename host_buffer<T>::const_iterator host_buffer<T>::begin() const
    {
        return _data.get();
    }

    template<typename T>
    typename host_buffer<T>::iterator host_buffer<T>::end()
    {
        return _data.get() + _size;
    }

    template<typename T>
    typename host_buffer<T>::const_iterator host_buffer<T>::end() const
    {
        return _data.get() + _size;
    }

    template<typename T>
    typename host_buffer<T>::reverse_iterator host_buffer<T>::rbegin()
    {
        return reverse_iterator { end() };
    }

    template<typename T>
    typename host_buffer<T>::const_reverse_iterator host_buffer<T>::rbegin() const
    {
        return const_reverse_iterator { end() };
    }

    template<typename T>
    typename host_buffer<T>::reverse_iterator host_buffer<T>::rend()
    {
        return reverse_iterator { begin() - 1 };
    }

    template<typename T>
    typename host_buffer<T>::const_reverse_iterator host_buffer<T>::rend() const
    {
        return const_reverse_iterator { begin() - 1 };
    }

    template<typename T>
    typename host_buffer<T>::size_type host_buffer<T>::size() const
    {
        return _size;
    }

    template<typename T>
    typename host_buffer<T>::size_type host_buffer<T>::nbytes() const
    {
        return _nbytes;
    }
}
