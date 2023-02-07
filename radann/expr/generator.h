#pragma once
#include "base.h"

namespace radann::expr
{
    template<typename Seq>
    class generator : public base<generator<Seq>>
    {
    public:
        using value_type = typename Seq::value_type;
        using strategy_type = meta::any_type;
        static constexpr bool is_expr = true;

    private:
        Seq _seq;

        generator(const Seq&);

    public:
        __host__ __device__ inline
        value_type operator[](size_t) const;

        size_t rank() const;
        auto shape() const;
        auto shape(size_t) const;

        template<typename _Seq>
        friend inline auto make_expr(const _Seq&);
    };
}

namespace radann::expr
{
    template<typename Seq>
    generator<Seq>::generator(const Seq &seq)
        : _seq(seq) {}

    template<typename Seq>
    __host__ __device__ inline
    typename generator<Seq>::value_type generator<Seq>::operator[](size_t i) const
    {
        return _seq(i);
    }

    template<typename Seq>
    size_t generator<Seq>::rank() const
    {
        return 0;
    }

    template<typename Seq>
    auto generator<Seq>::shape() const
    {
        return core::make_shape();
    }

    template<typename Seq>
    auto generator<Seq>::shape(size_t) const
    {
        throw std::invalid_argument("Index out of bounds.");
    }

    template<typename Seq>
    inline auto make_expr(const Seq &seq)
    {
        return generator {seq };
    }
}
