#pragma once
#include "base.h"

namespace radann::expr
{
    template<typename Seq>
    class term : public base<term<Seq>>
    {
    public:
        using value_type = typename Seq::value_type;
        static constexpr size_t rank = 0;
        static constexpr bool is_expr = true;
        static constexpr bool is_autodiff = false;

    private:
        Seq _seq;

        term(const Seq&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        auto shape() const;
        auto shape(size_t) const;

        template<typename Seq>
        friend inline auto make_term(const Seq&);
    };
}

namespace radann::expr
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
    auto term<Seq>::shape(size_t i) const
    {
        return make_shape()[i];
    }

    template<typename Seq>
    inline auto make_term(const Seq &seq)
    {
        return term { seq };
    }


}
