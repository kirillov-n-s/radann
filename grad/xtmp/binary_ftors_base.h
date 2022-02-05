#pragma once
#include "expr.h"

namespace grad::xtmp
{
    struct add
    {
        template <typename T>
        GRAD_DEVICE inline T operator()(T x, T y) const
        {
            return x + y;
        }
    };

    struct sub
    {
        template <typename T>
        GRAD_DEVICE inline T operator()(T x, T y) const
        {
            return x - y;
        }
    };

    struct mul
    {
        template <typename T>
        GRAD_DEVICE inline T operator()(T x, T y) const
        {
            return x * y;
        }
    };

    struct div
    {
        template <typename T>
        GRAD_DEVICE inline T operator()(T x, T y) const
        {
            return x / y;
        }
    };

    struct pow
    {
        template <typename T>
        GRAD_DEVICE inline T operator()(T x, T y) const
        {
            return ::pow(x, y);
        }
    };

    struct atan2
    {
        template <typename T>
        GRAD_DEVICE inline T operator()(T x, T y) const
        {
            return ::atan2(x, y);
        }
    };
    
    struct min
    {
        template <typename T>
        GRAD_DEVICE inline T operator()(T x, T y) const
        {
            return ::fmin(x, y);
        }
    };
    
    struct max
    {
        template <typename T>
        GRAD_DEVICE inline T operator()(T x, T y) const
        {
            return ::fmax(x, y);
        }
    };
}
