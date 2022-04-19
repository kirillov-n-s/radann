#pragma once

namespace grad
{
#if defined(GRAD_DEFAULT_DOUBLE)
    using real = double;
#else
    using real = float;
#endif
}
