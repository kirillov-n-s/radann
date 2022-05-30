#pragma once
#include <random>
#include "../expr/term.h"
#include "../func/random.h"

namespace radann
{
    template<typename T>
    inline auto uniform(unsigned int = std::random_device{}());

    template<typename T>
    inline auto normal(unsigned int = std::random_device{}());
}

namespace radann
{
    template<typename T>
    inline auto uniform(unsigned int seed)
    {
        return expr::make_term(func::uniform<T> {seed });
    }

    template<typename T>
    inline auto normal(unsigned int seed)
    {
        return expr::make_term(func::normal<T> {seed });
    }
}
