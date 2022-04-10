#pragma once
#include "cuda_runtime.h"

namespace grad::cuda
{
    template <typename T>
    class device_buffer
    {
    public:
        using value_type = T;

    private:
        T* _data;
        size_t _size;
        size_t _nbytes;

    public:
        device_buffer(size_t);
        device_buffer(const T*, size_t);
        ~device_buffer();

        T& operator[](size_t);
        const T& operator[](size_t) const;

        T* data();
        const T* data() const;

        size_t size() const;
        size_t nbytes() const;

        void copy_from(const T*, size_t);
        void copy_to(T*, size_t) const;
    };
}

namespace grad::cuda
{
    template<typename T>
    device_buffer<T>::device_buffer(size_t size)
        : _size(size), _nbytes(size * sizeof(T))
    {
        auto status = cudaMallocManaged(&_data, _nbytes);
        cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::bad_alloc();
    }

    template<typename T>
    device_buffer<T>::device_buffer(const T *device_src, size_t size)
        : device_buffer(size)
    {
        copy_from(device_src, size);
    }

    template<typename T>
    device_buffer<T>::~device_buffer()
    {
        cudaFree(_data);
    }

    template<typename T>
    T &device_buffer<T>::operator[](size_t i)
    {
        return _data[i];
    }

    template<typename T>
    const T &device_buffer<T>::operator[](size_t i) const
    {
        return _data[i];
    }

    template<typename T>
    T *device_buffer<T>::data()
    {
        return _data;
    }

    template<typename T>
    const T *device_buffer<T>::data() const
    {
        return _data;
    }

    template<typename T>
    size_t device_buffer<T>::size() const
    {
        return _size;
    }

    template<typename T>
    size_t device_buffer<T>::nbytes() const
    {
        return _nbytes;
    }

    template<typename T>
    void device_buffer<T>::copy_from(const T *device_src, size_t size)
    {
        auto status = cudaMemcpy(_data, device_src, size * sizeof(T),
                                 cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        if (status != cudaSuccess)
            throw std::runtime_error("device_buffer: copy_from failed. CUDA error status " + std::to_string(status));
    }

    template<typename T>
    void device_buffer<T>::copy_to(T *device_dst, size_t count) const
    {
        if (count > _size)
            throw std::runtime_error("Requested number of elements to copy is greater than buffer size.");
        auto status = cudaMemcpy(device_dst, _data, count * sizeof(T),
                                 cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        if (status != cudaSuccess)
            throw std::runtime_error("device_buffer: copy_to failed. CUDA error status " + std::to_string(status));
    }
}
