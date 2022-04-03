#include "grad/grad.h"
#include <chrono>

using timer = std::chrono::system_clock;

int main()
{
    const size_t k = 3;
    size_t n[k] = { 10, 100, 1000 };

    uint64_t time_alloc[k] = { 0 };
    uint64_t time_pure[k]  = { 0 };

    size_t tests = 1000;

    auto global_then = timer::now();
    for (int t = 0; t < tests; t++)
        for (int i = 0; i < k; i++)
        {
            auto m = n[i];

            auto then_alloc = timer::now();
            auto a = grad::make_array(3._fC / grad::asin(grad::arithm(0.f, 1.f)), m, m);
            auto b = grad::make_array(grad::pow(2._fC, grad::log(grad::arithm(0.f, -1.f))), m, m);

            auto then_pure = timer::now();
            auto c = grad::matmul(a, b);

            auto now = timer::now();
            time_alloc[i] += std::chrono::duration_cast<std::chrono::microseconds>(now - then_alloc).count();
            time_pure[i]  += std::chrono::duration_cast<std::chrono::microseconds>(now - then_pure).count();
        }

    std::cout << "tests = " << tests << ", k = " << k << "\n\n"
              << "full tests time = "
              << std::chrono::duration_cast<std::chrono::seconds>(timer::now() - global_then).count()
              << " s\n\n";
    for (int i = 0; i < k; i++)
        std::cout << "n = " << n[i] << '\n'
                  << "\tavg time with alloc  = " << time_alloc[i] / tests << " us\n"
                  << "\tavg time pure matmul = " << time_pure[i]  / tests << " us\n";

    std::cin.get();
}
