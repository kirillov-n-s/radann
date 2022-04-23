#pragma once

namespace grad
{
#if defined(GRAD_DEFAULT_REAL_DOUBLE)
    using real = double;
#else
    using real = float;
#endif

#if defined(GRAD_DEFAULT_AUTODIFF_FALSE)
    constexpr bool autodiff = false;
#else
    constexpr bool autodiff = true;
#endif
}
