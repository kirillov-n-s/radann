#pragma once
#include <random>
#include "../engine/term.h"
#include "../functor/random.h"

namespace radar
{
    template<typename T>
    inline auto uniform(unsigned int = std::random_device{}());

    template<typename T>
    inline auto normal(unsigned int = std::random_device{}());
}

namespace radar
{
    template<typename T>
    inline auto uniform(unsigned int seed)
    {
        return engine::make_term(functor::uniform<T> { seed });
    }

    template<typename T>
    inline auto normal(unsigned int seed)
    {
        return engine::make_term(functor::normal<T> { seed });
    }
}
