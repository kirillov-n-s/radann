#include "grad/grad.h"
#include <chrono>

using timer = std::chrono::system_clock;

int main()
{
    const size_t k10 = 8;
    size_t n10[k10] = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000 };

    const size_t k2 = 9;
    size_t n2[k2] = { 8, 64, 512, 4096, 32768, 262144, 2097152, 16777216, 134217728 };

    const size_t k_ = 10;
    size_t n_[k_] = { 32, 100, 128, 1000, 3000, 4096, 8192, 10000, 32768, 66000 };

    const auto k = k10;
    const auto n = n10;

    for (int i = 0; i < k; i++)
    {
        auto m = n[i];
        auto x = grad::make_arithm(grad::make_shape(m), 0.f, 1.f);
        auto f = grad::pow(2._fC, grad::log(x));

        auto r = grad::make_zeroes<float>(grad::make_shape());
        grad::cuda::reduce(x.data(), r.data(), x.size(), grad::functor::add{});

        std::cout << "\tsize = " << x.size() << '\n' << r;
    }

    /*uint64_t time[k]  = { 0 };

    size_t tests = 1000;

    auto global_then = timer::now();
    for (int t = 0; t < tests; t++)
        for (int i = 0; i < k; i++)
        {
            auto m = n[i];
            auto x = grad::make_arithm(grad::make_shape(m), 0.f, 1.f);
            auto f = grad::pow(2._fC, grad::log(x));

            auto then = timer::now();
            auto r = grad::make_zeroes<float>(grad::make_shape());
            auto y = grad::make_array(f);
            grad::cuda::reduce(y.data(), x.data(), x.size(), grad::functor::add{});
            time[i]  += std::chrono::duration_cast<std::chrono::microseconds>(timer::now() - then).count();
        }
    auto global_time = std::chrono::duration_cast<std::chrono::seconds>(timer::now() - global_then).count();

    std::cout << "tests = " << tests << ", k = " << k << "\n\n"
              << "full tests time = " << global_time << " s\n\n";
    for (int i = 0; i < k; i++)
        std::cout << "n = " << n[i] << '\n'
                  << "\tavg time reduce (sum) = " << time[i]  / tests << " us\n";*/

    std::cin.get();
}
