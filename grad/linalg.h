#pragma once
#include "eager_eval.h"
#include "cuda/cublas.h"

namespace grad
{
    template <typename Lhs, typename Rhs>
    inline auto dot(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Arg>
    inline auto norm2(const engine::expr<Arg>&);

    template <typename Lhs, typename Rhs>
    inline auto outer(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Lhs, typename Rhs>
    inline auto matmul(const engine::expr<Lhs>&, const engine::expr<Rhs>&);

    template <typename Arg>
    inline auto transpose(const engine::expr<Arg>&);

    /*template <typename Arg>
    inline auto inverse(const engine::expr<Arg>&);*/
}

namespace grad
{
    template <typename Lhs, typename Rhs>
    inline auto dot(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        using Tl = typename Lhs::value_type;
        using Tr = typename Rhs::value_type;
        static_assert(std::is_same_v<Tl, Tr>, "Type mismatch in dot product.");

        if (lhs.self().shape() != rhs.self().shape())
            throw std::invalid_argument("Shape mismatch in dot product.");

        auto lhs_eval = eager_eval(lhs);
        auto rhs_eval = eager_eval(rhs);

        auto res = make_array_of_zeroes<Tl>();
        cuda::cublas::dot(lhs_eval.data(), rhs_eval.data(),
                          res.data(),
                          lhs_eval.size());
        return res;
    }

    template <typename Arg>
    inline auto norm2(const engine::expr<Arg>& arg)
    {
        auto eval = eager_eval(arg);
        auto res = make_array_of_zeroes<Arg::value_type>();
        cuda::cublas::nrm2(eval.data(),
                           res.data(),
                           eval.size());
        return res;
    }

    template <typename Lhs, typename Rhs>
    inline auto outer(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        using Tl = typename Lhs::value_type;
        using Tr = typename Rhs::value_type;
        static_assert(std::is_same_v<Tl, Tr>, "Type mismatch in ger product.");
        static_assert(lhs.self().rank == 1 && rhs.self().rank == 1, "Rank mismatch in ger product.");

        auto lhs_eval = eager_eval(lhs);
        auto rhs_eval = eager_eval(rhs);

        auto rows = lhs_eval.size();
        auto cols = rhs_eval.size();
        auto res = make_array_of_zeroes<Tl>(rows, cols);
        cuda::cublas::ger(lhs_eval.data(), rhs_eval.data(),
                          res.data(),
                          rows, cols);
        return res;
    }

    template <typename Lhs, typename Rhs>
    inline auto matmul(const engine::expr<Lhs>& lhs, const engine::expr<Rhs>& rhs)
    {
        using Tl = typename Lhs::value_type;
        using Tr = typename Rhs::value_type;
        static_assert(std::is_same_v<Tl, Tr>, "Type mismatch in matrix multiplication.");

        const auto& lhs_self = lhs.self();
        const auto& rhs_self = rhs.self();

        constexpr auto lhs_rank = lhs_self.rank;
        constexpr auto rhs_rank = rhs_self.rank;
        static_assert(lhs_rank == 2 && (rhs_rank == 2 || rhs_rank == 1), "Rank mismatch in matrix multiply.");

        auto rows = lhs_self.shape(0);
        auto mid = lhs_self.shape(1);
        auto cols = rhs_self.shape(1);

        if (mid != rhs_self.shape(0))
            throw std::invalid_argument("Shape mismatch in matrix multiplication.");

        auto lhs_eval = eager_eval(lhs);
        auto rhs_eval = eager_eval(rhs);

        auto res = make_array_of_zeroes<Tl>(rows, cols);
        if constexpr(rhs_rank == 1)
            cuda::cublas::gemv(lhs_eval.data(), rhs_eval.data(),
                                res.data(),
                                rows, mid);
        else
            cuda::cublas::gemm(lhs_eval.data(), rhs_eval.data(),
                               res.data(),
                               rows, mid, cols);
        return res;
    }

    template <typename Arg>
    inline auto transpose(const engine::expr<Arg>& arg)
    {
        constexpr auto rank = arg.self().rank;
        static_assert(rank == 2 || rank == 1, "Rank mismatch in matrix transposition.");

        auto eval = eager_eval(arg);

        auto rows = eval.shape(0);
        auto cols = eval.shape(1);
        auto res = make_array_of_zeroes<Arg::value_type>(cols, rows);
        cuda::cublas::geam(eval.data(),
                           res.data(),
                           rows, cols);
        return res;
    }
}
