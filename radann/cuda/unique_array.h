#pragma once

namespace radann::cuda
{
    template<typename T>
    class unique_array
    {
    protected:
        T* _data;
        size_t _size;

    public:
        unique_array(size_t);
        ~unique_array();

        const T* data() const;
        T* data();

        size_t size() const;
    };
}

namespace radann::cuda
{
    template<typename T>
    unique_array<T>::unique_array(size_t size)
        : _size(size)
    {
        cudaMalloc(&_data, _size * sizeof(T));
        cudaMemset(_data, 0, _size * sizeof(T));
        auto status = cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::bad_alloc();
    }

    template<typename T>
    unique_array<T>::~unique_array()
    {
        cudaFree(_data);
    }

    template<typename T>
    const T *unique_array<T>::data() const
    {
        return _data;
    }

    template<typename T>
    T *unique_array<T>::data()
    {
        return _data;
    }

    template<typename T>
    size_t unique_array<T>::size() const
    {
        return _size;
    }
}
