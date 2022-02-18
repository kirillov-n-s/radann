#pragma once
#include "cuda_runtime.h"

namespace grad::cuda
{
    template <typename T>
    class storage
    {
    private:
        T* _data;
        size_t _size;
        size_t _nrefs = 1;

        storage(size_t);
        storage(const T*, size_t);

        ~storage();

    public:
        void copy(const T*, size_t = 0);

        void add_ref();
        void remove_ref();

        size_t size() const;
        size_t nrefs() const;

        T* data(size_t = 0);
        const T* data(size_t = 0) const;

        template<typename T>
        friend storage<T>* make_storage(size_t);
        template<typename T>
        friend storage<T>* make_storage(const T*, size_t);
    };
}

namespace grad::cuda
{
    template<typename T>
    storage<T>::storage(size_t size)
        : _size(size * sizeof(T))
    {
        auto status = cudaMallocManaged(&_data, _size);
        cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::bad_alloc();
    }

    template<typename T>
    storage<T>::storage(const T *src, size_t size)
        : _size(size * sizeof(T))
    {
        auto status = cudaMallocManaged(&_data, _size);
        cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::bad_alloc();
        copy(src);
    }

    template<typename T>
    storage<T>::~storage()
    {
        cudaFree(_data);
    }

    template<typename T>
    void storage<T>::copy(const T *src, size_t offset)
    {
        auto status = cudaMemcpy(_data + offset, src, _size, cudaMemcpyKind::cudaMemcpyDefault);
        cudaDeviceSynchronize();
        if (status != cudaError_t::cudaSuccess)
            throw std::bad_alloc();
    }

    template<typename T>
    void storage<T>::add_ref()
    {
        _nrefs++;
    }

    template<typename T>
    void storage<T>::remove_ref()
    {
        _nrefs--;
        if (_nrefs == 0)
            delete this;
    }

    template<typename T>
    size_t storage<T>::size() const
    {
        return _size;
    }

    template<typename T>
    size_t storage<T>::nrefs() const
    {
        return _nrefs;
    }

    template<typename T>
    T *storage<T>::data(size_t offset)
    {
        return _data + offset;
    }

    template<typename T>
    const T *storage<T>::data(size_t offset) const
    {
        return _data + offset;
    }

    template<typename T>
    storage<T> *make_storage(size_t size)
    {
        return new storage<T> { size };
    }

    template<typename T>
    storage<T> *make_storage(const T *src, size_t size)
    {
        return new storage<T> { src, size };
    }
}
