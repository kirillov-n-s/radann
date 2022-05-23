#pragma once
#include "default.h"
#include "../engine/tape_context.h"

namespace radann
{
    template<typename T = real>
    void reverse();

    template<typename T = real>
    void clear();

    template<typename T = real>
    void pause();
}

namespace radann
{
    template<typename T>
    void reverse()
    {
        radann::engine::get_tape<T>()->reverse();
    }

    template<typename T>
    void clear()
    {
        radann::engine::get_tape<T>()->clear();
    }

    template<typename T>
    void pause()
    {

    }
}
