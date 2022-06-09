#pragma once
#include "entry.h"

namespace radann::diff
{
    class policy_no_ad : public entry
    {
    public:
        using entry_type = entry;
        using entry_type::index_type;
        static constexpr bool does_record = false;

    protected:
        template<typename... Types>
        policy_no_ad(Types...) {}
    };
}
