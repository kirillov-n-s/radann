#pragma once
#include "../core/eval.h"

namespace grad::engine
{
    template<typename T, size_t NArg, size_t N>
    class unary_eager
    {
    public:
        using value_type = T;
        static constexpr size_t rank_arg = NArg;
        static constexpr size_t rank_result = N;

    private:
        array<NArg, T> _arg;
        array<N, T> _res;

        unary_eager(const array<NArg, T>&, const array<N, T>&);

    public:
        auto arg() const;
        auto result() const;

        template<typename Op, typename Arg>
        friend auto make_eager(const Op&, const expr<Arg>&);
    };
}

namespace grad::engine
{
    template<typename T, size_t NArg, size_t N>
    unary_eager<T, NArg, N>::unary_eager(const array<NArg, T> &arg, const array<N, T> &res)
        : _arg(arg), _res(res) {}

    template<typename T, size_t NArg, size_t N>
    auto unary_eager<T, NArg, N>::arg() const
    {
        return _arg;
    }

    template<typename T, size_t NArg, size_t N>
    auto unary_eager<T, NArg, N>::result() const
    {
        return _res;
    }

    template<typename Op, typename Arg>
    auto make_eager(const Op &op, const expr<Arg> &arg)
    {
        if constexpr(Op::requires_validation)
            op.validate(arg);
        auto arg_eval = eval(arg);
        return unary_eager {arg_eval, op(arg_eval) };
    }
}
