#pragma once
#include <type_traits>
#define GRAD_DEVICE __device__

namespace grad::xtmp
{
    template <typename Expr>
    class expr
    {
    public:
        inline const Expr &self() const
        {
            return static_cast<const Expr&>(*this);
        }
    };

    template<typename Op, typename Arg>
    class unary_expr : public expr<unary_expr<Op, Arg>>
    {
    public:
        using value_type = typename Arg::value_type;
        static const size_t rank = Arg::rank;

    private:
        const Op& _op;
        const Arg& _arg;

    public:
        unary_expr(const Op& op, const Arg& arg)
            : _op(op), _arg(arg) {}

        GRAD_DEVICE inline value_type operator[](size_t i) const
        {
            return _op(_arg[i]);
        }

        shape<rank> shape() const
        {
            return _arg.shape();
        }
    };

    template<typename Op, typename Lhs, typename Rhs>
    class binary_expr : public expr<binary_expr<Op, Lhs, Rhs>>
    {
    public:
        using value_type = std::common_type_t<typename Lhs::value_type, typename Rhs::value_type>;
        static const size_t rank = Lhs::rank;

    private:
        const Op& _op;
        const Lhs& _lhs;
        const Rhs& _rhs;

    public:
        binary_expr(const Op& op, const Lhs& lhs, const Rhs& rhs)
            : _op(op), _lhs(lhs), _rhs(rhs) {}

        GRAD_DEVICE inline value_type operator[](size_t i) const
        {
            return _op(_lhs[i], _rhs[i]);
        }

        shape<rank> shape() const
        {
            return _lhs.shape();
        }
    };

    template<typename Op, typename Arg>
    inline unary_expr<Op, Arg> make_unary_expr(const Op& op, const expr<Arg>& arg)
    {
        return { op, arg.self() };
    }

    template<typename Op, typename Lhs, typename Rhs>
    inline binary_expr<Op, Lhs, Rhs> make_binary_expr(const Op& op, const expr<Lhs>& lhs, const expr<Rhs>& rhs)
    {
        return { op, lhs.self(), rhs.self() };
    }
}
