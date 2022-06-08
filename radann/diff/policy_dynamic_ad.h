#pragma once
#include <optional>
#include "propagate.h"

namespace radann::diff
{
    class record_dynamic_ad
    {
    private:
        std::optional<size_t> _index;

    public:
        record_dynamic_ad(const std::optional<size_t>& = std::nullopt);

        bool ad() const;
        const std::optional<size_t>& grad_index() const;
    };

    class policy_dynamic_ad : public record_dynamic_ad
    {
    public:
        using record_type = record_dynamic_ad;

    protected:
        template<typename T>
        policy_dynamic_ad(bool);

    public:



    };
}
