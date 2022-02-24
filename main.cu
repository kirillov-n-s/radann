#include "grad/grad.h"
#include <chrono>

using timer = std::chrono::system_clock;

int main()
{
    size_t n = 10;

    grad::array<float, 1> x { grad::make_shape(n), grad::arithm(1.f, 1.f) };
    grad::array<float, 1> y { grad::make_shape(n), grad::arithm(-1.f, -1.f) };
    grad::array<float, 0> z { grad::make_shape() };

    std::cout << x << y << z << '\n';
    grad::cuda::linalg<float>::dot(x.data(), y.data(), z.data(), n);
    std::cout << x << y << z;

    std::cin.get();
}
