#pragma once
#include "../core/array.h"
#include "../diff/strategy_dynamic_ad.h"

namespace radann
{
    template<typename T = real>
    using array = core::array<T, diff::strategy_dynamic_ad<T>>;

    using core::shape;
    using core::make_shape;
    using core::save;

    using namespace func;
}
