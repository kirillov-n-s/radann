#pragma once
#include <optional>

namespace radann::diff
{
    class entry
    {
    public:
        using index_type = std::optional<size_t>;

    private:
        index_type _index;

    public:
        entry(const index_type& = std::nullopt);

        bool ad() const;
        const index_type& grad_index() const;
        void set_grad_index(const index_type&);
    };
}

namespace radann::diff
{
    entry::entry(const index_type &index)
        : _index(index)
    {}

    bool entry::ad() const
    {
        return _index.has_value();
    }

    const entry::index_type &entry::grad_index() const
    {
        return _index;
    }

    void entry::set_grad_index(const entry::index_type &other)
    {
        _index = other;
    }
}
