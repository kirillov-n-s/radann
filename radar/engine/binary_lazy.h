#pragma once
#include "access.h"

namespace radar::engine
{
    template<typename Op, typename Lhs, typename Rhs>
    class binary_lazy : public expr<binary_lazy<Op, Lhs, Rhs>>
    {
    public:
        using value_type = std::common_type_t<typename Lhs::value_type, typename Rhs::value_type>;
        static constexpr size_t rank = std::max(Lhs::rank, Rhs::rank);
        static constexpr bool is_expr = true;
        static constexpr bool is_autodiff = Lhs::is_autodiff || Rhs::is_autodiff;

    private:
        Op _op;
        Lhs _lhs;
        Rhs _rhs;

    public:
        binary_lazy(const Op&, const Lhs&, const Rhs&);

        __host__ __device__ inline
        value_type operator[](size_t) const;

        auto shape() const;
        size_t shape(size_t) const;

        template<typename Expr>
        void propagate_grad(const expr<Expr>&) const;

        template<typename Op, typename Lhs, typename Rhs>
        friend inline auto make_lazy(const Op&, const expr<Lhs>&, const expr<Rhs>&);
    };
}

namespace radar::engine
{
    template<typename Op, typename Lhs, typename Rhs>
    engine::binary_lazy<Op, Lhs, Rhs>::binary_lazy(const Op &op, const Lhs &lhs, const Rhs &rhs)
        : _op(op), _lhs(lhs), _rhs(rhs) {}

    template<typename Op, typename Lhs, typename Rhs>
    __host__ __device__
    typename binary_lazy<Op, Lhs, Rhs>::value_type binary_lazy<Op, Lhs, Rhs>::operator[](size_t i) const
    {
        return _op(_lhs[i], _rhs[i]);
    }

    template<typename Op, typename Lhs, typename Rhs>
    auto binary_lazy<Op, Lhs, Rhs>::shape() const
    {
        if constexpr(Lhs::rank > Rhs::rank)
            return _lhs.shape();
        else
            return _rhs.shape();
    }

    template<typename Op, typename Lhs, typename Rhs>
    size_t binary_lazy<Op, Lhs, Rhs>::shape(size_t i) const
    {
        return (Lhs::rank > Rhs::rank) ? _lhs.shape(i) : _rhs.shape(i);
    }

    template<typename Op, typename Lhs, typename Rhs>
    template<typename Expr>
    void binary_lazy<Op, Lhs, Rhs>::propagate_grad(const expr<Expr> &mult) const
    {
        if constexpr (Lhs::is_autodiff)
            _lhs.propagate_grad(_op.accumulate_grad_lhs(_lhs, _rhs, mult));
        if constexpr (Rhs::is_autodiff)
            _rhs.propagate_grad(_op.accumulate_grad_rhs(_lhs, _rhs, mult));
    }

    template<typename Op, typename Lhs, typename Rhs>
    inline auto make_lazy(const Op& op, const expr<Lhs>& lhs, const expr<Rhs>& rhs)
    {
        return binary_lazy {op, get_access(lhs.self()), get_access(rhs.self()) };
    }
}
