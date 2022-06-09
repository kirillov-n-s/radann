#pragma once
#include "../core/array.h"
#include "../diff/policy_dynamic_ad.h"

namespace radann
{
    template<typename T = real>
    using array = core::array<T, diff::policy_dynamic_ad<T>>;

    using core::shape;
    using core::make_shape;

    using namespace func;
}
