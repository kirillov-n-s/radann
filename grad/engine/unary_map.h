#pragma once
#include "../core/eval.h"

namespace grad::engine
{
    template<typename T, size_t N, size_t R>
    class unary_map
    {
    public:
        using value_type = T;
        static constexpr size_t rank_arg = N;
        static constexpr size_t rank_result = R;

    private:
        array<T, N> _arg;
        array<T, R> _res;

        unary_map(const array<T, N>&, const array<T, R>&);

    public:
        auto arg() const;
        auto result() const;

        template<typename Op, typename Arg>
        friend auto make_map(const Op&, const expr<Arg>&);
    };
}

namespace grad::engine
{
    template<typename T, size_t N, size_t R>
    unary_map<T, N, R>::unary_map(const array<T, N> &arg, const array<T, R> &res)
        : _arg(arg), _res(res) {}

    template<typename T, size_t N, size_t R>
    auto unary_map<T, N, R>::arg() const
    {
        return _arg;
    }

    template<typename T, size_t N, size_t R>
    auto unary_map<T, N, R>::result() const
    {
        return _res;
    }

    template<typename Op, typename Arg>
    auto make_map(const Op &op, const expr<Arg> &arg)
    {
        if constexpr(Op::requires_validation)
            op.validate(arg);
        auto arg_eval = eval(arg);
        return unary_map { arg_eval, op(arg_eval) };
    }
}
