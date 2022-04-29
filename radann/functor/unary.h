#pragma once

namespace radann::functor
{
    struct neg
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return -x;
        }
    };

    struct abs
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::fabs(x);
        }
    };

    struct sqrt
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::sqrt(x);
        }
    };

    struct cbrt
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::cbrt(x);
        }
    };

    struct sin
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::sin(x);
        }
    };

    struct cos
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::cos(x);
        }
    };

    struct tan
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::tan(x);
        }
    };

    struct asin
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::asin(x);
        }
    };

    struct acos
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::acos(x);
        }
    };
    
    struct atan
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::atan(x);
        }
    };
    
    struct sinh
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::sinh(x);
        }
    };

    struct cosh
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::cosh(x);
        }
    };
    
    struct tanh
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::tanh(x);
        }
    };

    struct asinh
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::asinh(x);
        }
    };
    
    struct acosh
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::acosh(x);
        }
    };

    struct atanh
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::atanh(x);
        }
    };

    struct exp
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::exp(x);
        }
    };

    struct exp2
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::exp2(x);
        }
    };

    struct exp10
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::exp10(x);
        }
    };

    struct expm1
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::expm1(x);
        }
    };

    struct log
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::log(x);
        }
    };

    struct log2
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::log2(x);
        }
    };

    struct log10
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::log10(x);
        }
    };

    struct log1p
    {
        template <typename T>
        __host__ __device__ inline
		T operator()(T x) const
        {
            return ::log1p(x);
        }
    };
}
