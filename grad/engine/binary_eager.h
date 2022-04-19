#pragma once
#include "../core/eval.h"

namespace grad::engine
{
    template<typename T, size_t NLhs, size_t NRhs, size_t N>
    class binary_eager
    {
    public:
        using value_type = T;
        static constexpr size_t rank_lhs = NLhs;
        static constexpr size_t rank_rhs = NRhs;
        static constexpr size_t rank_result = N;

    private:
        array<NLhs, T> _lhs;
        array<NRhs, T> _rhs;
        array<N, T> _res;

        binary_eager(const array<NLhs, T>&, const array<NRhs, T>&, const array<N, T>&);

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
    template<typename T, size_t NLhs, size_t NRhs, size_t N>
    binary_eager<T, NLhs, NRhs, N>::binary_eager(
            const array<NLhs, T> &lhs, const array<NRhs, T> &rhs, const array<N, T> &res)
        : _lhs(lhs), _rhs(rhs), _res(res) {}

    template<typename T, size_t NLhs, size_t NRhs, size_t N>
    auto binary_eager<T, NLhs, NRhs, N>::lhs() const
    {
        return _lhs;
    }

    template<typename T, size_t NLhs, size_t NRhs, size_t N>
    auto binary_eager<T, NLhs, NRhs, N>::rhs() const
    {
        return _rhs;
    }

    template<typename T, size_t NLhs, size_t NRhs, size_t N>
    auto binary_eager<T, NLhs, NRhs, N>::result() const
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
        return binary_eager {lhs_eval, rhs_eval, op(lhs_eval, rhs_eval) };
    }
}
