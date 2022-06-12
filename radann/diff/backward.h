#pragma once
#include "../func/linalg.h"
#include "strategy_no_ad.h"

namespace radann::diff
{
    template<typename T>
    using array_no_ad = core::array<T, strategy_no_ad>;

    template<typename T>
    using backward_function = void(*)(array_no_ad<T>&, const array_no_ad<T>&, const array_no_ad<T>&);

    struct backward_default
    {
        using backward_lhs = backward_default;
        using backward_rhs = backward_default;
    };

    template<typename Tag>
    struct backward
    {
        template<typename T>
        static void function(array_no_ad<T>& dx, const array_no_ad<T>& dy, const array_no_ad<T>& mult)
        {
            dx += mult * dy;
        }
    };

    template<>
    struct backward<typename core::matmul<false, false>::backward_lhs>
    {
        template<typename T>
        static void function(array_no_ad<T>& dx, const array_no_ad<T>& dy, const array_no_ad<T>& mult)
        {
            dx += func::matmul<false, true>(dy, mult);
        }
    };

    template<>
    struct backward<typename core::matmul<false, false>::backward_rhs>
    {
        template<typename T>
        static void function(array_no_ad<T>& dx, const array_no_ad<T>& dy, const array_no_ad<T>& mult)
        {
            dx += func::matmul<true, false>(mult, dy);
        }
    };
}
