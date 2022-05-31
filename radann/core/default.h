#pragma once

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
}
