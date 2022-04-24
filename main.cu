#include "grad/grad.h"
#include <chrono>

using timer = std::chrono::system_clock;

int main()
{
    auto x = grad::make_arithm(grad::make_shape(3, 4, 2, 3), 1.f, 1.f);
    auto y = x.reshape(grad::make_shape(8, 9));
    auto z = x.flatten();
    auto w = x(1);

    auto dx = x.grad();
    auto dy = y.grad();
    auto dz = z.grad();
    auto dw = w.grad();

    std::cout << x << y << z << w;
    std::cout << "\n-------------GRADIENTS-------------\n\n\n";
    std::cout << dx << dy << dz << dw;

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
            auto x = grad::make_arithm(grad::make_shape(m), 0.f, 1.f);
            auto y = grad::make_array(grad::sigmoid(x));

            auto then = timer::now();
            auto z = grad::eval(grad::sin(x) / grad::pow2(y) + get_grad::log(3._fC));
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
