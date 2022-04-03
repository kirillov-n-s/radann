#pragma once
#include "access.h"

namespace grad::engine
{
    template<typename Op, typename Lhs, typename Rhs>
    class binary : public expr<binary<Op, Lhs, Rhs>>
    {
    public:
        using value_type = std::common_type_t<typename Lhs::value_type, typename Rhs::value_type>;
        static constexpr size_t rank = std::max(Lhs::rank, Rhs::rank);
        static constexpr bool is_expr = true;

    private:
        Op _op;
        Lhs _lhs;
        Rhs _rhs;

        binary(const Op&, const Lhs&, const Rhs&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        auto shape() const;

        template<typename Op, typename Lhs, typename Rhs>
        friend inline auto make_binary(const Op&, const expr<Lhs>&, const expr<Rhs>&);
    };
}

namespace grad::engine
{
    template<typename Op, typename Lhs, typename Rhs>
    engine::binary<Op, Lhs, Rhs>::binary(const Op &op, const Lhs &lhs, const Rhs &rhs)
        : _op(op), _lhs(lhs), _rhs(rhs) {}

    template<typename Op, typename Lhs, typename Rhs>
    __host__ __device__
    typename binary<Op, Lhs, Rhs>::value_type binary<Op, Lhs, Rhs>::operator[](size_t i) const
    {
        return _op(_lhs[i], _rhs[i]);
    }

    template<typename Op, typename Lhs, typename Rhs>
    auto binary<Op, Lhs, Rhs>::shape() const
    {
        if constexpr(Lhs::rank > Rhs::rank)
            return _lhs.shape();
        else
            return _rhs.shape();
    }

    template<typename Op, typename Lhs, typename Rhs>
    inline auto make_binary(const Op& op, const expr<Lhs>& lhs, const expr<Rhs>& rhs)
    {
        return binary { op, get_access(lhs.self()), get_access(rhs.self()) };
    }
}
