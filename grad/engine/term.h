#pragma once
#include "expr.h"

namespace grad::engine
{
    template<typename Seq>
    class term : public expr<term<Seq>>
    {
    public:
        using value_type = typename Seq::value_type;
        static constexpr size_t rank = 0;
        static constexpr bool is_expr = true;

    private:
        Seq _seq;

        term(const Seq&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        auto shape() const;

        template<typename Seq>
        friend inline auto make_term(const Seq&);
    };
}

namespace grad::engine
{
    template<typename Seq>
    term<Seq>::term(const Seq &seq)
        : _seq(seq) {}

    template<typename Seq>
    __host__ __device__ inline
    typename term<Seq>::value_type term<Seq>::operator[](size_t i) const
    {
        return _seq(i);
    }

    template<typename Seq>
    auto term<Seq>::shape() const
    {
        return make_shape();
    }

    template<typename Seq>
    inline auto make_term(const Seq &seq)
    {
        return term { seq };
    }
}
