#include "grad/grad.h"
#include <chrono>

using timer = std::chrono::system_clock;

int main()
{
    /*auto ilist = { 1.f, 3.f, 3.f, 7.f, 2.f, 2.f, 8.f, 6.f, 9.f, 4.f, 2.f, 0.f, 1.f, 4.f, 8.f, 8.f };

    auto x = grad::make_array(grad::make_shape(ilist.size()), ilist);
    auto y = x.reshape(grad::make_shape(4, 4));
    auto w = y.reshape(grad::make_shape(4, 2, 2));
    auto u = w.reshape(grad::make_shape(2, 2, 2, 2));
    auto v = u.reshape(grad::make_shape(1, 16));

    std::cout << x << y << w << u << v;*/

    const size_t k = 11;
    size_t n[k] = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };

    //uint64_t time_alloc[k] = { 0 };
    uint64_t time_pure[k]  = { 0 };

    size_t s = 0;

    size_t tests = 1000;

    auto global_then = timer::now();
    for (int t = 0; t < tests; t++)
        for (int i = 0; i < k; i++)
        {
            auto m = n[i];
            auto x = grad::make_arithm(grad::make_shape(m, m), 0.f, 1.f);

            //auto then_alloc = timer::now();
            auto a = 3._fC / grad::asin(x);
            auto b = grad::pow(2._fC, grad::log(x));

            auto then_pure = timer::now();
            auto c = grad::matmul(a, b);

            s += c.size();

            auto now = timer::now();
            //time_alloc[i] += std::chrono::duration_cast<std::chrono::microseconds>(now - then_alloc).count();
            time_pure[i]  += std::chrono::duration_cast<std::chrono::microseconds>(now - then_pure).count();
        }

    std::cout << "tests = " << tests << ", k = " << k << "\n\n"
              << "full tests time = "
              << std::chrono::duration_cast<std::chrono::seconds>(timer::now() - global_then).count()
              << " s\n\n";
    for (int i = 0; i < k; i++)
        std::cout << "n = " << n[i] << '\n'
                  //<< "\tavg time with alloc  = " << time_alloc[i] / tests << " us\n"
                  << "\tavg time matmul = " << time_pure[i]  / tests << " us\n";

    std::cout << "\n" << s;

    std::cin.get();
}
