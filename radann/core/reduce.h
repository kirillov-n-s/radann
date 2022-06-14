#pragma once
#include "../cuda/reduce.h"
//#include "../cuda/cublas.h"
#include "array.h"
#include "../oper/binary.h"

namespace radann::core
{
    struct sum
    {
        static constexpr bool does_validate = false;

        template <typename T, typename Strategy>
        auto operator()(const array<T, Strategy> &x) const
        {
            auto res = array<T, Strategy> { make_shape(), x.ad() };
            cuda::reduce(x.data(), res.data(), x.size(), oper::add{});
            return res;
        }
    };
}
