#pragma once
#include <optional>

namespace radann::diff
{
    struct record_no_ad
    {
        bool ad() const;
        const std::optional<size_t>& grad_index() const;
    };

    struct policy_no_ad : public record_no_ad
    {
        using record_type = record_no_ad;
    };
}
