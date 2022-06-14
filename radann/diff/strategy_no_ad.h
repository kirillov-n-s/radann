#pragma once
#include "entry.h"

namespace radann::diff
{
    class strategy_no_ad : public entry
    {
    public:
        using entry_type = entry;
        static constexpr bool does_record = false;
        static constexpr bool does_link = false;

    protected:
        template<typename... Types>
        strategy_no_ad(Types...) {}
    };
}
