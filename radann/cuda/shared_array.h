#pragma once
#include "shared_storage.h"

namespace radann::cuda
{
    template<typename T>
    class shared_array
    {
    protected:
        cuda::shared_storage<T>* _storage;
        size_t _size;
        size_t _offset = 0;

        void copy(const host_buffer<T>&, size_t = 0);
        void link(shared_storage<T>*);

    public:
        shared_array(size_t);
        shared_array(const T*, size_t);
        shared_array(shared_storage<T>*, size_t, size_t);
        shared_array(const shared_array&);

        ~shared_array();

        void zero();

        const T* data() const;
        T* data();

        cuda::shared_storage<T>* storage() const;
        cuda::host_buffer<T> host() const;

        size_t size() const;
        size_t offset() const;
    };
}

namespace radann::cuda
{
    template<typename T>
    void shared_array<T>::copy(const host_buffer<T> &host, size_t offset)
    {
        _storage->copy(host, offset);
    }

    template<typename T>
    void shared_array<T>::link(shared_storage<T> *storage)
    {
        _storage->remove_ref();
        _storage = storage;
        _storage->add_ref();
    }

    template<typename T>
    shared_array<T>::shared_array(size_t size)
        : _storage(cuda::make_storage<T>(size)), _size(size)
    {}

    template<typename T>
    shared_array<T>::shared_array(const T *device_ptr, size_t size)
        : _storage(cuda::make_storage(device_ptr, size)), _size(size)
    {}

    template<typename T>
    shared_array<T>::shared_array(shared_storage<T> *storage, size_t size, size_t offset)
        : _storage(storage), _size(size), _offset(offset)
    {
        _storage->add_ref();
    }

    template<typename T>
    shared_array<T>::shared_array(const shared_array<T> &other)
        : shared_array(other._storage, other._size, other._offset)
    {}

    template<typename T>
    shared_array<T>::~shared_array()
    {
        _storage->remove_ref();
    }

    template<typename T>
    void shared_array<T>::zero()
    {
        _storage->zero(_size, _offset);
    }

    template<typename T>
    const T *shared_array<T>::data() const
    {
        return _storage->data(_offset);
    }

    template<typename T>
    T *shared_array<T>::data()
    {
        return _storage->data(_offset);
    }

    template<typename T>
    shared_storage<T> *shared_array<T>::storage() const
    {
        return _storage;
    }

    template<typename T>
    host_buffer<T> shared_array<T>::host() const
    {
        return _storage->host(_size, _offset);
    }

    template<typename T>
    size_t shared_array<T>::size() const
    {
        return _size;
    }

    template<typename T>
    size_t shared_array<T>::offset() const
    {
        return _offset;
    }
}
