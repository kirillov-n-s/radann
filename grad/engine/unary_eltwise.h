#pragma once
#include "access.h"
#include "unary_map.h"


namespace grad::engine
{
    template<typename Op, typename Arg>
    class unary_eltwise : public expr<unary_eltwise<Op, Arg>>
    {
    public:
        using value_type = typename Arg::value_type;
        static constexpr size_t rank = Arg::rank;
        static constexpr bool is_expr = true;

    private:
        Op _op;
        Arg _arg;

        unary_eltwise(const Op&, const Arg&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        auto shape() const;
        auto shape(size_t) const;

        template<typename Op, typename Arg>
        friend inline auto make_eltwise(const Op&, const expr<Arg>&);
    };
}

namespace grad::engine
{
    template<typename Op, typename Arg>
    unary_eltwise<Op, Arg>::unary_eltwise(const Op &op, const Arg &arg)
        : _op(op), _arg(arg) {}

    template<typename Op, typename Arg>
    __host__ __device__
    typename unary_eltwise<Op, Arg>::value_type unary_eltwise<Op, Arg>::operator[](size_t i) const
    {
        return _op(_arg[i]);
    }

    template<typename Op, typename Arg>
    auto unary_eltwise<Op, Arg>::shape() const
    {
        return _arg.shape();
    }

    template<typename Op, typename Arg>
    auto unary_eltwise<Op, Arg>::shape(size_t i) const
    {
        return _arg.shape(i);
    }

    template<typename Op, typename Arg>
    inline auto make_eltwise(const Op& op, const expr<Arg>& arg)
    {
        return unary_eltwise {op, get_access(arg.self()) };
    }


}
