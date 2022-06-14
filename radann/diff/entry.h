#pragma once

namespace radann::diff
{
    using index_t = int64_t;

    class entry
    {
    public:
        using index_type = index_t;
        static constexpr index_type null_index = -1;

    protected:
        index_type _index;

    public:
        entry(index_type = null_index);

        bool ad() const;
        index_type grad_index() const;
    };
}

namespace radann::diff
{
    entry::entry(index_type index)
        : _index(index)
    {}

    bool entry::ad() const
    {
        return _index != null_index;
    }

    entry::index_type entry::grad_index() const
    {
        return _index;
    }
}
