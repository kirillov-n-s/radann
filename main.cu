#include "radann/radann.h"
#include <chrono>
#include <iostream>

using timer = std::chrono::system_clock;

int main()
{
    auto s = radann::make_shape(1);

    auto x0 = radann::make_constant(s, 1.337f);
    auto x1 = radann::make_constant(s, 1.488f);

    auto y = radann::make_constant(s, 4.f);
    auto z = radann::eval(2._fC * x0 + 3._fC * x1 * x1);
    y *= radann::sin(z);

    y.set_grad(2._fC);

    radann::reverse();

    std::cout << "x0 =\n" << x0 << "dx0 =\n" << x0.get_grad()
              << "x1 =\n" << x1 << "dx1 =\n" << x1.get_grad()
              << "y =\n" << y  << "dy =\n" << y.get_grad();

    radann::clear();

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
