#pragma once
#include "host_buffer.h"

namespace radann::cuda
{
    template <typename T>
    class shared_storage
    {
    private:
        T* _data;
        size_t _size;
        size_t _nrefs = 1;

        shared_storage(size_t);
        shared_storage(const T*, size_t);

        ~shared_storage();

    public:
        shared_storage(const shared_storage&) = delete;

        void copy(const T*, size_t, size_t = 0);
        void copy(const host_buffer<T>&, size_t = 0);

        void zero(size_t, size_t);

        void add_ref();
        void remove_ref();

        size_t size() const;
        size_t nrefs() const;

        T* data(size_t = 0);
        const T* data(size_t = 0) const;

        host_buffer<T> host(size_t, size_t = 0) const;

        template<typename _T>
        friend shared_storage<_T>* make_storage(size_t);
        template<typename _T>
        friend shared_storage<_T>* make_storage(const _T*, size_t);
    };
}

namespace radann::cuda
{
    template<typename T>
    shared_storage<T>::shared_storage(size_t size)
        : _size(size)
    {
        cudaMalloc(&_data, _size * sizeof(T));
        zero(_size, 0);
    }

    template<typename T>
    shared_storage<T>::shared_storage(const T *device_ptr, size_t size)
        : _size(size)
    {
        cudaMalloc(&_data, _size * sizeof(T));
        copy(device_ptr, size);
    }

    template<typename T>
    shared_storage<T>::~shared_storage()
    {
        cudaFree(_data);
    }

    template<typename T>
    void shared_storage<T>::copy(const T *device_ptr, size_t size, size_t offset)
    {
        cudaMemcpy(_data + offset, device_ptr, size * sizeof(T),
                   cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        auto status = cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::bad_alloc();
    }

    template<typename T>
    void shared_storage<T>::copy(const host_buffer<T> &host, size_t offset)
    {
        cudaMemcpy(_data + offset, host.data(), host.size() * sizeof(T),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);
        auto status = cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::bad_alloc();
    }

    template<typename T>
    void shared_storage<T>::zero(size_t size, size_t offset)
    {
        cudaMemset(_data + offset, 0, size * sizeof(T));
        auto status = cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::bad_alloc();
    }

    template<typename T>
    void shared_storage<T>::add_ref()
    {
        _nrefs++;
    }

    template<typename T>
    void shared_storage<T>::remove_ref()
    {
        if (--_nrefs == 0)
            delete this;
    }

    template<typename T>
    size_t shared_storage<T>::size() const
    {
        return _size;
    }

    template<typename T>
    size_t shared_storage<T>::nrefs() const
    {
        return _nrefs;
    }

    template<typename T>
    T *shared_storage<T>::data(size_t offset)
    {
        return _data + offset;
    }

    template<typename T>
    const T *shared_storage<T>::data(size_t offset) const
    {
        return _data + offset;
    }

    template<typename T>
    host_buffer<T> shared_storage<T>::host(size_t size, size_t offset) const
    {
        return host_buffer<T> { _data + offset, size };
    }

    template<typename T>
    shared_storage<T> *make_storage(size_t size)
    {
        return new shared_storage<T> {size };
    }

    template<typename T>
    shared_storage<T> *make_storage(const T *device_ptr, size_t size)
    {
        return new shared_storage<T> {device_ptr, size };
    }
}
