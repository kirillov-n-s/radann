#pragma once
#include "access.h"

namespace grad::engine
{
    template<typename Op, typename Arg>
    class unary : public expr<unary<Op, Arg>>
    {
    public:
        using value_type = typename Arg::value_type;
        static constexpr size_t rank = Arg::rank;
        static constexpr bool is_expr = true;

    private:
        Op _op;
        Arg _arg;

        unary(const Op&, const Arg&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        auto shape() const;
        auto shape(size_t) const;

        template<typename Op, typename Arg>
        friend inline auto make_unary(const Op&, const expr<Arg>&);
    };
}

namespace grad::engine
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
    inline auto make_unary(const Op& op, const expr<Arg>& arg)
    {
        return unary { op, get_access(arg.self()) };
    }
}
