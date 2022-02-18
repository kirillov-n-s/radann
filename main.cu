#include "grad/grad.h"

int main()
{
    grad::array<float, 1> x
    {
        grad::make_shape(150),
        grad::arithm(0.f, 0.1f)
    };

    grad::array<float, 1> f = grad::sqrt(x) + grad::sin(x);
    grad::array<float, 1> g = f + grad::normal<float>() / 3._fC;
    grad::array<float, 1> h = f + grad::uniform<float>() - 0.5_fC;

    std::cout << "BEFORE\n\n" << x << f << g << h;

    f += grad::array<float, 1> (grad::make_shape(10), 2._fC);
    g *= grad::constant(-1.f);
    h = g - 2._fC;

    std::cout << "\n\nAFTER\n\n" << x << f << g << h;

    std::cin.get();
}
