#pragma once
#include "access.h"

namespace radann::expr
{
    template<typename Op, typename Arg>
    class unary : public base<unary<Op, Arg>>
    {
    public:
        using value_type = typename Arg::value_type;
        static constexpr bool is_expr = true;

    private:
        Op _op;
        Arg _arg;

        unary(const Op&, const Arg&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        size_t rank() const;
        auto shape() const;
        auto shape(size_t) const;

        bool ad() const;
        template<typename Expr>
        void propagate_grad(const base<Expr>&) const;

        template<typename Op, typename Arg>
        friend inline auto make_expr(const Op&, const base<Arg>&);
    };
}

namespace radann::expr
{
    template<typename Op, typename Arg>
    unary<Op, Arg>::unary(const Op &op, const Arg &arg)
        : _op(op), _arg(arg) {}

    template<typename Op, typename Arg>
    __host__ __device__
    typename unary<Op, Arg>::value_type unary<Op, Arg>::operator[](size_t i) const
    {
        return _op(_arg[i]);
    }

    template<typename Op, typename Arg>
    size_t unary<Op, Arg>::rank() const
    {
        return _arg.rank();
    }

    template<typename Op, typename Arg>
    auto unary<Op, Arg>::shape() const
    {
        return _arg.shape();
    }

    template<typename Op, typename Arg>
    auto unary<Op, Arg>::shape(size_t i) const
    {
        return _arg.shape(i);
    }

    template<typename Op, typename Arg>
    bool unary<Op, Arg>::ad() const
    {
        return _arg.ad();
    }

    template<typename Op, typename Arg>
    template<typename Expr>
    void unary<Op, Arg>::propagate_grad(const base<Expr> &mult) const
    {
        if constexpr (Arg::is_autodiff)
            _arg.propagate_grad(_op.accumulate_grad(_arg, mult));
    }

    template<typename Op, typename Arg>
    inline auto make_expr(const Op& op, const base<Arg>& arg)
    {
        return unary {op, get_access(arg.self()) };
    }
}
