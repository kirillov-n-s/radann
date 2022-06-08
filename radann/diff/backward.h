#pragma once
#include "../core/array.h"
#include "policy_no_ad.h"

namespace radann::diff
{
    template<typename T>
    using array_no_ad = core::array<T, policy_no_ad>;

    template<typename T>
    using backward_function = void(*)(array_no_ad<T>, const array_no_ad<T>, const array_no_ad<T>);

    template<typename Op>
    struct backward
    {
        template<typename T>
        static void function(array_no_ad<T> output, const array_no_ad<T> input, const array_no_ad<T> mult)
        {
            output += mult * input;
        }
    };
}
