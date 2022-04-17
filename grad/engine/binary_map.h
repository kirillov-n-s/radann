#pragma once
#include "../core/eval.h"

namespace grad::engine
{
    template<typename T, size_t NLhs, size_t NRhs, size_t R>
    class binary_map
    {
    public:
        using value_type = T;
        static constexpr size_t rank_lhs = NLhs;
        static constexpr size_t rank_rhs = NRhs;
        static constexpr size_t rank_result = R;

    private:
        array<T, NLhs> _lhs;
        array<T, NRhs> _rhs;
        array<T, R> _res;

        binary_map(const array<T, NLhs>&, const array<T, NRhs>&, const array<T, R>&);

    public:
        auto lhs() const;
        auto rhs() const;
        auto result() const;

        template <typename Op, typename Lhs, typename Rhs>
        friend auto make_map(const Op&, const expr<Lhs>&, const expr<Rhs>&);
    };
}

namespace grad::engine
{
    template<typename T, size_t NLhs, size_t NRhs, size_t R>
    binary_map<T, NLhs, NRhs, R>::binary_map(const array<T, NLhs> &lhs, const array<T, NRhs> &rhs, const array<T, R> &res)
        : _lhs(lhs), _rhs(rhs), _res(res) {}

    template<typename T, size_t NLhs, size_t NRhs, size_t R>
    auto binary_map<T, NLhs, NRhs, R>::lhs() const
    {
        return _lhs;
    }

    template<typename T, size_t NLhs, size_t NRhs, size_t R>
    auto binary_map<T, NLhs, NRhs, R>::rhs() const
    {
        return _rhs;
    }

    template<typename T, size_t NLhs, size_t NRhs, size_t R>
    auto binary_map<T, NLhs, NRhs, R>::result() const
    {
        return _res;
    }

    template<typename Op, typename Lhs, typename Rhs>
    auto make_map(const Op &op, const expr<Lhs> &lhs, const expr<Rhs> &rhs)
    {
        if constexpr(Op::requires_validation)
            op.validate(lhs, rhs);
        auto lhs_eval = eval(lhs);
        auto rhs_eval = eval(rhs);
        return binary_map { lhs_eval, rhs_eval, op(lhs_eval, rhs_eval) };
    }
}
