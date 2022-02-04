#pragma once
#include "thrust/copy.h"

namespace grad::kernel
{
    template <typename InputIterator, typename OutputIterator>
    OutputIterator copy(InputIterator, InputIterator, OutputIterator);
}

namespace grad::kernel
{
    template <typename InputIterator, typename OutputIterator>
    OutputIterator copy(InputIterator first, InputIterator last, OutputIterator dst)
    {
        return thrust::copy(first, last, dst);
    }
}
