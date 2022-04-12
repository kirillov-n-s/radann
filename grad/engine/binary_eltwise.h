#pragma once
#include "access.h"

namespace grad::engine
{
    template<typename Op, typename Lhs, typename Rhs>
    class binary_eltwise : public expr<binary_eltwise<Op, Lhs, Rhs>>
    {
    public:
        using value_type = std::common_type_t<typename Lhs::value_type, typename Rhs::value_type>;
        static constexpr size_t rank = std::max(Lhs::rank, Rhs::rank);
        static constexpr bool is_expr = true;

    private:
        Op _op;
        Lhs _lhs;
        Rhs _rhs;

        binary_eltwise(const Op&, const Lhs&, const Rhs&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        auto shape() const;
        auto shape(size_t) const;

        template<typename Op, typename Lhs, typename Rhs>
        friend inline auto make_eltwise(const Op&, const expr<Lhs>&, const expr<Rhs>&);
    };
}

namespace grad::engine
{
    template<typename Op, typename Lhs, typename Rhs>
    engine::binary_eltwise<Op, Lhs, Rhs>::binary_eltwise(const Op &op, const Lhs &lhs, const Rhs &rhs)
        : _op(op), _lhs(lhs), _rhs(rhs) {}

    template<typename Op, typename Lhs, typename Rhs>
    __host__ __device__
    typename binary_eltwise<Op, Lhs, Rhs>::value_type binary_eltwise<Op, Lhs, Rhs>::operator[](size_t i) const
    {
        return _op(_lhs[i], _rhs[i]);
    }

    template<typename Op, typename Lhs, typename Rhs>
    auto binary_eltwise<Op, Lhs, Rhs>::shape() const
    {
        if constexpr(Lhs::rank > Rhs::rank)
            return _lhs.shape();
        else
            return _rhs.shape();
    }

    template<typename Op, typename Lhs, typename Rhs>
    auto binary_eltwise<Op, Lhs, Rhs>::shape(size_t i) const
    {
        if constexpr(Lhs::rank > Rhs::rank)
            return _lhs.shape(i);
        else
            return _rhs.shape(i);
    }

    template<typename Op, typename Lhs, typename Rhs>
    inline auto make_eltwise(const Op& op, const expr<Lhs>& lhs, const expr<Rhs>& rhs)
    {
        return binary_eltwise {op, get_access(lhs.self()), get_access(rhs.self()) };
    }
}
