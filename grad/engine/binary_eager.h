#pragma once
#include "../core/eval.h"

namespace grad::engine
{
    template<size_t NLhs, size_t NRhs, size_t N,
             bool ADLhs, bool ADRhs, bool AD,
             typename T>
    class binary_eager
    {
    public:
        using value_type = T;
        static constexpr size_t rank_lhs = NLhs;
        static constexpr size_t rank_rhs = NRhs;
        static constexpr size_t rank_result = N;

    private:
        array<NLhs, ADLhs, T> _lhs;
        array<NRhs, ADRhs, T> _rhs;
        array<N, AD, T> _res;

        binary_eager(const array<NLhs, ADLhs, T>&, const array<NRhs, ADRhs, T>&, const array<N, AD, T>&);

    public:
        auto lhs() const;
        auto rhs() const;
        auto result() const;

        template <typename Op, typename Lhs, typename Rhs>
        friend auto make_eager(const Op&, const expr<Lhs>&, const expr<Rhs>&);
    };
}

namespace grad::engine
{
    template<size_t NLhs, size_t NRhs, size_t N,
             bool ADLhs, bool ADRhs, bool AD,
             typename T>
    binary_eager<NLhs, NRhs, N, ADLhs, ADRhs, AD, T>::binary_eager(
            const array<NLhs, ADLhs, T> &lhs, const array<NRhs, ADRhs, T> &rhs, const array<N, AD, T> &res)
        : _lhs(lhs), _rhs(rhs), _res(res) {}

    template<size_t NLhs, size_t NRhs, size_t N,
             bool ADLhs, bool ADRhs, bool AD,
             typename T>
    auto binary_eager<NLhs, NRhs, N, ADLhs, ADRhs, AD, T>::lhs() const
    {
        return _lhs;
    }

    template<size_t NLhs, size_t NRhs, size_t N,
             bool ADLhs, bool ADRhs, bool AD,
             typename T>
    auto binary_eager<NLhs, NRhs, N, ADLhs, ADRhs, AD, T>::rhs() const
    {
        return _rhs;
    }

    template<size_t NLhs, size_t NRhs, size_t N,
             bool ADLhs, bool ADRhs, bool AD,
             typename T>
    auto binary_eager<NLhs, NRhs, N, ADLhs, ADRhs, AD, T>::result() const
    {
        return _res;
    }

    template<typename Op, typename Lhs, typename Rhs>
    auto make_eager(const Op &op, const expr<Lhs> &lhs, const expr<Rhs> &rhs)
    {
        if constexpr(Op::requires_validation)
            op.validate(lhs, rhs);
        auto lhs_eval = eval(lhs);
        auto rhs_eval = eval(rhs);
        return binary_eager<lhs_eval.rank, rhs_eval.rank, decltype(op(lhs_eval, rhs_eval))::rank,
                            lhs_eval.is_autodiff, rhs_eval.is_autodiff, decltype(op(lhs_eval, rhs_eval))::is_autodiff,
                            typename decltype(op(lhs_eval, rhs_eval))::value_type>
            { lhs_eval, rhs_eval, op(lhs_eval, rhs_eval) };
    }
}
