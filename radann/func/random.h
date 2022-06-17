#pragma once
#include <random>
#include "../expr/generator.h"
#include "../oper/random.h"

namespace radann::func
{
    template<typename T>
    inline auto uniform(unsigned int = std::random_device{}());

    template<typename T>
    inline auto normal(unsigned int = std::random_device{}());
}

namespace radann::func
{
    template<typename T>
    inline auto uniform(unsigned int seed)
    {
        return expr::make_expr(oper::uniform<T>{seed});
    }

    template<typename T>
    inline auto normal(unsigned int seed)
    {
        return expr::make_expr(oper::normal<T>{seed});
    }
}
