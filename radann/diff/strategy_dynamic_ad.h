#pragma once
#include "propagate.h"
#include "entry.h"

namespace radann::diff
{
    template<typename T>
    class strategy_dynamic_ad : public entry
    {
    public:
        using entry_type = entry;
        using entry_type::index_type;
        static constexpr bool does_record = true;

    private:
        index_type::value_type grad_index_value() const;

    protected:
        strategy_dynamic_ad();
        strategy_dynamic_ad(const core::shape&, bool);
        strategy_dynamic_ad(const core::shape&, size_t, const index_type&, bool);

        template<typename Tag = backward_default, typename Expr>
        void record_grad(const expr::base<Expr>&) const;

    public:
        core::array<T, strategy_dynamic_ad<T>> get_grad() const;
        template<typename Expr>
        void set_grad(const expr::base<Expr>&) const;
        void set_grad() const;
    };
}

namespace radann::diff
{
    template<typename T>
    strategy_dynamic_ad<T>::index_type::value_type strategy_dynamic_ad<T>::grad_index_value() const
    {
        return grad_index().value();
    }

    template<typename T>
    strategy_dynamic_ad<T>::strategy_dynamic_ad()
        : entry_type()
    {}

    template<typename T>
    strategy_dynamic_ad<T>::strategy_dynamic_ad(const core::shape &shape, bool ad)
        : entry_type(ad
                ? index_type { diff::get_tape<T>()->create_grad(shape) }
                : index_type {})
    {}

    template<typename T>
    strategy_dynamic_ad<T>::strategy_dynamic_ad(const core::shape &shape, size_t offset,
                                                const index_type &base_index, bool derive)
        : entry_type(!base_index.has_value()
                ? index_type {}
                : (derive
                   ? index_type { diff::get_tape<T>()->derive_grad(base_index.value(), shape, offset) }
                   : base_index))
    {}

    template<typename T>
    template<typename Tag, typename Expr>
    void strategy_dynamic_ad<T>::record_grad(const expr::base<Expr> &expr) const
    {
        auto expr_self = expr.self();
        if (is_ad(expr_self) && ad())
        {
            propagate<Tag>(expr_self, func::constant<T>(1));
            get_tape<T>()->push_lvalue(grad_index_value());
        }
    }

    template<typename T>
    core::array<T, strategy_dynamic_ad<T>> strategy_dynamic_ad<T>::get_grad() const
    {
        return core::array<T, strategy_dynamic_ad<T>> {get_tape<T>()->get_grad(grad_index_value()) };
    }

    template<typename T>
    template<typename Expr>
    void strategy_dynamic_ad<T>::set_grad(const expr::base<Expr> &expr) const
    {
        get_tape<T>()->set_grad(grad_index_value(), expr);
    }

    template<typename T>
    void strategy_dynamic_ad<T>::set_grad() const
    {
        set_grad(func::constant<T>(1));
    }
}
