#include "policy_dynamic_ad.h"

namespace radann::diff
{
    record_dynamic_ad::record_dynamic_ad(const std::optional<size_t> &index)
        : _index(index)
    {}

    bool record_dynamic_ad::ad() const
    {
        return _index.has_value();
    }

    const std::optional<size_t> &record_dynamic_ad::grad_index() const
    {
        return _index;
    }

    template<typename T>
    policy_dynamic_ad::policy_dynamic_ad(bool ad)
    {}
}
