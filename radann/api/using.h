#pragma once
#include "../core/array.h"
#include "../diff/policy_dynamic_ad.h"

namespace radann
{

#if defined(RADANN_DEFAULT_REAL_DOUBLE)
    using real = double;
#else
    using real = float;
#endif

#if defined(RADANN_DEFAULT_AUTODIFF_FALSE)
    constexpr bool autodiff = false;
#else
    constexpr bool autodiff = true;
#endif

    template<typename T = real>
    using array = core::array<T, diff::policy_dynamic_ad>;

    using core::shape;
    using core::make_shape;

    using namespace func;
}
