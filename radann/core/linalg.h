#pragma once
#include "../cuda/cublas.h"
#include "array.h"

namespace radann::core
{
    template<bool LTrans, bool RTrans>
    struct matmul
    {
        struct backward_lhs {};
        struct backward_rhs {};

        static constexpr bool does_validate = true;

        template<typename Lhs, typename Rhs>
        void validate(const expr::base<Lhs> &lhs, const expr::base<Rhs> &rhs) const
        {
            auto lself = lhs.self();
            auto rself = rhs.self();

            auto lrank = lself.rank();
            auto rrank = rself.rank();

            auto lcols = lself.shape(!LTrans);
            auto rrows = rself.shape(RTrans);

            if (lrank > 2 || rrank > lrank || rrank < 1 || lcols != rrows)
                throw std::invalid_argument("Illegal shape in matrix multiplication.");
        }

        template <typename T, typename Strategy>
        auto operator()(const array<T, Strategy> &x, const array<T, Strategy> &y) const
        {
            auto xrank = x.rank();
            auto yrank = y.rank();

            if (xrank > yrank)
            {
                auto res = array<T, Strategy> { make_shape(x.shape(LTrans)) };
                cuda::cublas::gemv<LTrans>(x.data(), y.data(), res.data(), x.shape(0), x.shape(1));
                return res;
            }

            if (xrank == 1)
            {
                if constexpr(RTrans)
                {
                    auto rows = x.size();
                    auto cols = y.size();
                    auto res = array<T, Strategy>{ make_shape(rows, cols) };
                    cuda::cublas::ger(x.data(), y.data(), res.data(), rows, cols);
                    return res;
                }
                else
                {
                    auto res = array<T, Strategy> { make_shape() };
                    cuda::cublas::dot(x.data(), y.data(), res.data(), x.size());
                    return res;
                }
            }

            auto rows = x.shape(LTrans);
            auto mid = x.shape(!LTrans);
            auto cols = y.shape(!RTrans);
            auto res = array<T, Strategy> { make_shape(rows, cols) };
            cuda::cublas::gemm<LTrans, RTrans>(x.data(), y.data(), res.data(), rows, mid, cols);
            return res;
        }
    };
}
