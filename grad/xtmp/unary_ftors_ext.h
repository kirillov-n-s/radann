#pragma once
#include "expr.h"

namespace grad::xtmp
{
    struct sgn
    {
        template <typename T>
        GRAD_DEVICE inline T operator()(T x) const
        {
            return x / ::fabs(x);
        }
    };
}
