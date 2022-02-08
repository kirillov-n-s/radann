#pragma once

namespace grad::ops
{
    struct add
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x, T y) const
        {
            return x + y;
        }
    };

    struct sub
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x, T y) const
        {
            return x - y;
        }
    };

    struct mul
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x, T y) const
        {
            return x * y;
        }
    };

    struct div
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x, T y) const
        {
            return x / y;
        }
    };

    struct pow
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x, T y) const
        {
            return ::pow(x, y);
        }
    };

    struct atan2
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x, T y) const
        {
            return ::atan2(x, y);
        }
    };
    
    struct min
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x, T y) const
        {
            return ::fmin(x, y);
        }
    };
    
    struct max
    {
        template <typename T>
        __host__ __device__ inline T operator()(T x, T y) const
        {
            return ::fmax(x, y);
        }
    };
}
