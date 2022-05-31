#pragma once
#include "base.h"

namespace radann::expr
{
    template<typename Seq>
    class term : public base<term<Seq>>
    {
    public:
        using value_type = typename Seq::value_type;
        static constexpr bool is_expr = true;

    private:
        Seq _seq;

        term(const Seq&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        size_t rank() const;
        auto shape() const;
        auto shape(size_t) const;

        bool ad() const;

        template<typename Seq>
        friend inline auto make_expr(const Seq&);
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
    size_t term<Seq>::rank() const
    {
        return 0;
    }

    template<typename Seq>
    auto term<Seq>::shape() const
    {
        return make_shape();
    }

    template<typename Seq>
    auto term<Seq>::shape(size_t) const
    {
        throw std::invalid_argument("Index out of bounds.");
    }

    template<typename Seq>
    inline auto make_expr(const Seq &seq)
    {
        return term { seq };
    }

    template<typename Seq>
    bool term<Seq>::ad() const
    {
        return false;
    }
}
