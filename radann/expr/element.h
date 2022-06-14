#pragma once
#include "base.h"

namespace radann::expr
{
    template<typename Seq>
    class element : public base<element<Seq>>
    {
    public:
        using value_type = typename Seq::value_type;
        using strategy_type = meta::any_type;
        static constexpr bool is_expr = true;

    private:
        Seq _seq;

        element(const Seq&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        size_t rank() const;
        auto shape() const;
        auto shape(size_t) const;

        template<typename Seq>
        friend inline auto make_expr(const Seq&);
    };
}

namespace radann::expr
{
    template<typename Seq>
    element<Seq>::element(const Seq &seq)
        : _seq(seq) {}

    template<typename Seq>
    __host__ __device__ inline
    typename element<Seq>::value_type element<Seq>::operator[](size_t i) const
    {
        return _seq(i);
    }

    template<typename Seq>
    size_t element<Seq>::rank() const
    {
        return 0;
    }

    template<typename Seq>
    auto element<Seq>::shape() const
    {
        return core::make_shape();
    }

    template<typename Seq>
    auto element<Seq>::shape(size_t) const
    {
        throw std::invalid_argument("Index out of bounds.");
    }

    template<typename Seq>
    inline auto make_expr(const Seq &seq)
    {
        return element {seq };
    }
}
