#pragma once
#include <type_traits>
#include "access.h"

namespace grad::ops
{
    template<typename Op, typename Lhs, typename Rhs>
    class binary : public expr<binary<Op, Lhs, Rhs>>
    {
    public:
        using value_type = std::common_type_t<typename Lhs::value_type, typename Rhs::value_type>;
        static const size_t rank = Lhs::rank;

    private:
        Op _op;
        Lhs _lhs;
        Rhs _rhs;

        binary(const Op&, const Lhs&, const Rhs&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        shape<rank> shape() const;

        template<typename Op, typename Lhs, typename Rhs>
        friend inline
        binary<Op, Lhs, Rhs> make_binary(const Op&, const expr<Lhs>&, const expr<Rhs>&);
    };
}

namespace grad::ops
{
    template<typename Op, typename Lhs, typename Rhs>
    ops::binary<Op, Lhs, Rhs>::binary(const Op &op, const Lhs &lhs, const Rhs &rhs)
        : _op(op), _lhs(lhs), _rhs(rhs) {}

    template<typename Op, typename Lhs, typename Rhs>
    __host__ __device__
    typename binary<Op, Lhs, Rhs>::value_type binary<Op, Lhs, Rhs>::operator[](size_t i) const
    {
        return _op(_lhs[i], _rhs[i]);
    }

    template<typename Op, typename Lhs, typename Rhs>
    shape<binary<Op, Lhs, Rhs>::rank> binary<Op, Lhs, Rhs>::shape() const
    {
        return _lhs.shape();
    }

    template<typename Op, typename Lhs, typename Rhs>
    inline binary<Op, Lhs, Rhs> make_binary(const Op& op, const expr<Lhs>& lhs, const expr<Rhs>& rhs)
    {
        return { op, lhs.self(), rhs.self() };
    }
}
