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
        static constexpr bool does_record = true;
        static constexpr bool does_link = true;

    protected:
        strategy_dynamic_ad();
        strategy_dynamic_ad(const core::shape&, bool);
        strategy_dynamic_ad(const core::shape&, size_t, index_type, bool);

        template<typename Tag = backward_default, typename Expr>
        void record_grad(const expr::base<Expr>&) const;
        void link_grad(index_type);

    public:
        ~strategy_dynamic_ad();
        core::array<T, strategy_dynamic_ad<T>> grad() const;
        void inactive_grad();
    };
}

namespace radann::diff
{
    template<typename T>
    strategy_dynamic_ad<T>::strategy_dynamic_ad()
        : entry_type()
    {}

    template<typename T>
    strategy_dynamic_ad<T>::strategy_dynamic_ad(const core::shape &shape, bool ad)
        : entry_type(ad
                     ? get_tape<T>()->create_grad(shape)
                     : null_index)
    {}

    template<typename T>
    strategy_dynamic_ad<T>::strategy_dynamic_ad(const core::shape &shape, size_t offset,
                                                index_type base_index, bool derive)
        : entry_type(base_index == null_index
                ? null_index
                : (derive
                   ? get_tape<T>()->derive_grad(base_index, shape, offset)
                   : get_tape<T>()->copy_grad(base_index)))
    {}

    template<typename T>
    template<typename Tag, typename Expr>
    void strategy_dynamic_ad<T>::record_grad(const expr::base<Expr> &expr) const
    {
        auto expr_self = expr.self();
        if (is_ad(expr_self) && ad())
        {
            propagate<Tag>(expr_self, func::constant<T>(1));
            get_tape<T>()->push_statement(_index);
        }
    }

    template<typename T>
    void strategy_dynamic_ad<T>::link_grad(index_type index)
    {
        inactive_grad();
        _index = index;
        if (ad())
            get_tape<T>()->copy_grad(_index);
    }

    template<typename T>
    strategy_dynamic_ad<T>::~strategy_dynamic_ad()
    {
        inactive_grad();
    }

    template<typename T>
    core::array<T, strategy_dynamic_ad<T>> strategy_dynamic_ad<T>::grad() const
    {
        if (!ad())
            throw std::runtime_error("Cannot get gradient of non-ad array.");
        return core::array<T, strategy_dynamic_ad<T>> {get_tape<T>()->get_grad(_index) };
    }

    template<typename T>
    void strategy_dynamic_ad<T>::inactive_grad()
    {
        if (ad())
            get_tape<T>()->delete_grad(_index);
        _index = null_index;
    }
}
