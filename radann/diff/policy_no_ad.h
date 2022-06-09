#pragma once
#include <optional>

namespace radann::diff
{
    struct entry_no_ad
    {
        using index_type = std::optional<size_t>;

        template<typename... Types>
        entry_no_ad(Types...) {}

        bool ad() const;
        const index_type& grad_index() const;
    };

    class policy_no_ad : public entry_no_ad
    {
    public:
        using entry_type = entry_no_ad;
        using entry_type::index_type;
        static constexpr bool has_record = false;

    protected:
        template<typename... Types>
        policy_no_ad(Types...) {}
    };
}

namespace radann::diff
{
    bool entry_no_ad::ad() const
    {
        return false;
    }

    const entry_no_ad::index_type &entry_no_ad::grad_index() const
    {
        return std::nullopt;
    }
}
