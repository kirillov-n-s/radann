#pragma once
#include "tape.h"

namespace radann::diff
{
    template<typename T>
    class tape_context
    {
    private:
        tape<T>* _handle;
        tape_context();

    public:
        tape_context(const tape_context&) = delete;
        ~tape_context();

        template<typename _T>
        friend tape<_T>* get_tape();
    };
}

namespace radann::diff
{
    template<typename T>
    tape_context<T>::tape_context()
        : _handle(new tape<T>()) {}

    template<typename T>
    tape_context<T>::~tape_context()
    {
        delete _handle;
    }

    template<typename T>
    tape<T> *get_tape()
    {
        static tape_context<T> context;
        return context._handle;
    }
}
