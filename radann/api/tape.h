#pragma once
#include "using.h"
#include "../diff/tape_context.h"

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
        radann::expr::get_tape<T>()->reverse();
    }

    template<typename T>
    void clear()
    {
        radann::expr::get_tape<T>()->clear();
    }

    template<typename T>
    void pause()
    {

    }
}
