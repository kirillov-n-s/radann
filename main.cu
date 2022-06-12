#include "radann/radann.h"
#include <chrono>

using timer = std::chrono::system_clock;

struct messy_identity
{
    static constexpr bool does_validate = false;

    template <typename T>
    auto operator()(const radann::array<T>& x) const
    {
        return x;
    }
};

namespace radann::diff
{
    template<typename Arg, typename Mult>
    auto grad(const radann::expr::base<Arg>&, const radann::expr::base<Mult>&, const messy_identity&)
    {
        return radann::constant<typename Arg::value_type>(1);
    }
}

template<>
struct radann::diff::backward<messy_identity>
{
    template<typename T>
    static void function(array_no_ad<T> &dx, const array_no_ad<T> &dy, const array_no_ad<T> &mult)
    {
        dx += mult * dy + radann::constant(0.1f);
    }
};

struct s : radann::diff::backward<messy_identity> {};

template <typename Arg>
inline auto mess(const radann::expr::base<Arg>& arg)
{
    return radann::core::eager(messy_identity{}, arg);
}

int main()
{
    /*auto s = radann::make_shape(2, 2);

    auto x0 = radann::make_constant(s, 1.337f);
    auto x1 = radann::make_constant(s, 1.488f);

    auto y = radann::make_constant(s, 4.f);
    auto z = radann::make_array(2._fC * x0 + 3._fC * x1 * x1);
    y *= radann::sin(z);

    y.set_grad(2._fC);

    radann::reverse();

    std::cout << "dx0 =\n" << x0.get_grad()
              << "dx1 =\n" << x1.get_grad()
              << "y =\n" << y;*/

    /*auto s = radann::make_shape(3, 3);

    auto x = radann::make_arithm(s, -4.f, 1.f);
    radann::array<> y = radann::sigmoid(x);
    y.set_grad();

    auto t = radann::copy(x);
    radann::array<> z = 1._fC / (1._fC + radann::exp(-t));
    z.set_grad();

    radann::reverse();

    std::cout << x << t
              << "\n----------------------\n\n"
              << y << z
              << "\n----------------------\n\n"
              << x.get_grad() << t.get_grad();*/

    /*auto a = radann::make_arithm(radann::make_shape(10), 1.f, 1.f);
    auto b = radann::make_geom(radann::make_shape(3, 3), 1.f, 2.f);
    auto c = radann::make_constant(radann::make_shape(2, 2, 2), 13.f);

    auto x = mess(a + 3._fC);
    auto y = mess(b / radann::constant<radann::real>(b.size()));
    auto z = mess(c * c);

    x.set_grad();
    y.set_grad();
    z.set_grad();

    radann::reverse();

    std::cout << "\n----------------INPUT------------------\n\n"
              << a << b << c
              << "\n----------------MESS-----------------\n\n"
              << x << y << z
              << "\n----------------GRAD-------------------\n\n"
              << a.get_grad() << b.get_grad() << c.get_grad();*/

    auto o = 3, i = 4, n = 5;

    auto w = radann::make_constant(radann::make_shape(o, i), 0.5f);
    auto x = radann::make_arithm(radann::make_shape(i), 1.f, 1.f);
    auto y = radann::matmul(w, x);

    y.set_grad();
    radann::reverse();

    std::cout << "W = \n" << w
              << "X = \n" << x
              << "Y = \n" << y
              << "dW = \n" << w.get_grad()
              << "dX = \n" << x.get_grad();

    std::cin.get();
}

/*int main()
{
    const size_t k10 = 8;
    size_t n10[k10] = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000 };

    const size_t k2 = 11;
    size_t n2[k2] = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };

    const auto k = k10;
    const auto n = n10;

    uint64_t time[k] = { 0 };

    size_t tests = 1000;

    auto global_then = timer::now();
    for (int t = 0; t < tests; t++)
        for (int i = 0; i < k; i++)
        {
            auto m = n[i];
            auto x = radann::make_arithm(radann::make_shape(m), 0.f, 1.f);
            auto y = radann::make_array(radann::sigmoid(x));

            auto then = timer::now();
            auto z = radann::eval(radann::sin(x) / radann::pow2(y) + get_grad::log(3._fC));
            time[i] += std::chrono::duration_cast<std::chrono::microseconds>(timer::now() - then).count();
        }
    auto global_time = std::chrono::duration_cast<std::chrono::seconds>(timer::now() - global_then).count();

    std::cout << "tests = " << tests << ", k = " << k << "\n\n"
              << "full tests time = " << global_time << " s\n\n";
    for (int i = 0; i < k; i++)
        std::cout << "n = " << n[i] << '\n'
                  << "\tavg time = " << time[i] / tests << " us\n";

    std::cin.get();
}*/
