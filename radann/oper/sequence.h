#pragma once

namespace radann::oper
{
    template <typename T>
    class constant
    {
    public:
        using value_type = T;

    private:
        T _value;

    public:
        constant(T value)
            : _value(value) {};

        __host__ __device__
        inline T operator()(size_t) const
        {
            return _value;
        }
    };

    template <typename T>
    class arithm
    {
    public:
        using value_type = T;

    private:
        T _offset, _step;

    public:
        arithm(T offset, T step)
            : _offset(offset), _step(step) {};

        __host__ __device__
        inline T operator()(size_t i) const
        {
            return _offset + i * _step;
        }
    };

    template <typename T>
    class geom
    {
    public:
        using value_type = T;

    private:
        T _scale, _ratio;

    public:
        geom(T scale, T ratio)
            : _scale(scale), _ratio(ratio) {};

        __host__ __device__
        inline T operator()(size_t i) const
        {
            return _scale * ::pow(_ratio, i);
        }
    };
}
