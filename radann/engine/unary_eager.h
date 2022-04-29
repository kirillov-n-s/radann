#pragma once
#include "../core/eval.h"

namespace radann::engine
{
    template<size_t NArg, size_t N,
             bool ADArg, bool AD,
             typename T>
    class unary_eager
    {
    public:
        using value_type = T;
        static constexpr size_t rank_arg = NArg;
        static constexpr size_t rank_result = N;

    private:
        array<NArg, ADArg, T> _arg;
        array<N, AD, T> _res;

        unary_eager(const array<NArg, ADArg, T>&, const array<N, AD, T>&);

    public:
        auto arg() const;
        auto result() const;

        template<typename Op, typename Arg>
        friend auto make_eager(const Op&, const expr<Arg>&);
    };
}

namespace radann::engine
{
    template<size_t NArg, size_t N, bool ADArg, bool AD, typename T>
    unary_eager<NArg, N, ADArg, AD, T>::unary_eager(const array<NArg, ADArg, T> &arg, const array<N, AD, T> &res)
        : _arg(arg), _res(res) {}

    template<size_t NArg, size_t N, bool ADArg, bool AD, typename T>
    auto unary_eager<NArg, N, ADArg, AD, T>::arg() const
    {
        return _arg;
    }

    template<size_t NArg, size_t N, bool ADArg, bool AD, typename T>
    auto unary_eager<NArg, N, ADArg, AD, T>::result() const
    {
        return _res;
    }

    template<typename Op, typename Arg>
    auto make_eager(const Op &op, const expr<Arg> &arg)
    {
        if constexpr(Op::requires_validation)
            op.validate(arg);
        auto arg_eval = eval(arg);
        return unary_eager<arg_eval.rank, decltype(op(arg_eval))::rank,
                           arg_eval.is_autodiff, decltype(op(arg_eval))::is_autodiff,
                           typename decltype(op(arg_eval))::value_type>
            {arg_eval, op(arg_eval) };
    }
}
