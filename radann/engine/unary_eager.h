#pragma once
#include "../core/eval.h"
#include "access.h"

namespace radann::engine
{
    template<typename Op,
             size_t NArg, size_t N,
             bool ADArg, bool AD,
             typename T>
    class unary_eager
    {
    public:
        using value_type = T;
        static constexpr size_t rank = N;
        static constexpr bool is_expr = true;
        static constexpr bool is_autodiff = AD;

    private:
        Op _op;

        array<NArg, ADArg, T> _arg;
        array<N, AD, T> _res;

        access<N, AD, T> _access;

        unary_eager(const Op&, const array<NArg, ADArg, T>&, const array<N, AD, T>&);

    public:
        auto arg() const;
        auto result() const;

        auto shape() const;
        auto shape(size_t) const;

        template<typename Expr>
        void propagate_grad(const expr<Expr>&) const;

        template<typename Op, typename Arg>
        friend auto make_eager(const Op&, const expr<Arg>&);
    };
}

namespace radann::engine
{
    template<typename Op, size_t NArg, size_t N, bool ADArg, bool AD, typename T>
    unary_eager<Op, NArg, N, ADArg, AD, T>::unary_eager(const Op &op,
                                                        const array<NArg, ADArg, T> &arg,
                                                        const array<N, AD, T> &res)
        : _op(op),
          _arg(arg), _res(res),
          _access(get_access(res))
    {}

    template<typename Op, size_t NArg, size_t N, bool ADArg, bool AD, typename T>
    auto unary_eager<Op, NArg, N, ADArg, AD, T>::arg() const
    {
        return _arg;
    }

    template<typename Op, size_t NArg, size_t N, bool ADArg, bool AD, typename T>
    auto unary_eager<Op, NArg, N, ADArg, AD, T>::result() const
    {
        return _res;
    }

    template<typename Op, size_t NArg, size_t N, bool ADArg, bool AD, typename T>
    auto unary_eager<Op, NArg, N, ADArg, AD, T>::shape() const
    {
        return _res.shape();
    }

    template<typename Op, size_t NArg, size_t N, bool ADArg, bool AD, typename T>
    auto unary_eager<Op, NArg, N, ADArg, AD, T>::shape(size_t i) const
    {
        return _res.shape(i);
    }

    template<typename Op, size_t NArg, size_t N, bool ADArg, bool AD, typename T>
    template<typename Expr>
    void unary_eager<Op, NArg, N, ADArg, AD, T>::propagate_grad(const expr<Expr> &mult) const
    {
        if constexpr (AD)
            _access.propagate_grad(_op.accumulate_grad(_arg, mult));
    }

    template<typename Op, typename Arg>
    auto make_eager(const Op &op, const expr<Arg> &arg)
    {
        if constexpr(Op::requires_validation)
            op.validate(arg);
        auto arg_eval = eval(arg);
        return unary_eager<Op, arg_eval.rank, decltype(op(arg_eval))::rank,
                           arg_eval.is_autodiff, decltype(op(arg_eval))::is_autodiff,
                           typename decltype(op(arg_eval))::value_type>
            {arg_eval, op(arg_eval) };
    }
}
