#include "policy_no_ad.h"

namespace radann::diff
{
    bool record_no_ad::ad() const
    {
        return false;
    }

    const std::optional <size_t> &record_no_ad::grad_index() const
    {
        return std::nullopt;
    }
}
