#pragma once
#include "access.h"

namespace radar::engine
{
    template<typename Op, typename Arg>
    class unary_lazy : public expr<unary_lazy<Op, Arg>>
    {
    public:
        using value_type = typename Arg::value_type;
        static constexpr size_t rank = Arg::rank;
        static constexpr bool is_expr = true;
        static constexpr bool is_autodiff = Arg::is_autodiff;

    private:
        Op _op;
        Arg _arg;

        unary_lazy(const Op&, const Arg&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        auto shape() const;
        auto shape(size_t) const;

        template<typename Expr>
        void propagate_grad(const expr<Expr>&) const;

        template<typename Op, typename Arg>
        friend inline auto make_lazy(const Op&, const expr<Arg>&);
    };
}

namespace radar::engine
{
    template<typename Op, typename Arg>
    unary_lazy<Op, Arg>::unary_lazy(const Op &op, const Arg &arg)
        : _op(op), _arg(arg) {}

    template<typename Op, typename Arg>
    __host__ __device__
    typename unary_lazy<Op, Arg>::value_type unary_lazy<Op, Arg>::operator[](size_t i) const
    {
        return _op(_arg[i]);
    }

    template<typename Op, typename Arg>
    auto unary_lazy<Op, Arg>::shape() const
    {
        return _arg.shape();
    }

    template<typename Op, typename Arg>
    auto unary_lazy<Op, Arg>::shape(size_t i) const
    {
        return _arg.shape(i);
    }

    template<typename Op, typename Arg>
    template<typename Expr>
    void unary_lazy<Op, Arg>::propagate_grad(const expr<Expr> &mult) const
    {
        if constexpr (Arg::is_autodiff)
            _arg.propagate_grad(_op.accumulate_grad(_arg, mult));
    }

    template<typename Op, typename Arg>
    inline auto make_lazy(const Op& op, const expr<Arg>& arg)
    {
        return unary_lazy {op, get_access(arg.self()) };
    }
}
