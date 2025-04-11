#ifndef ACCURATE_MATH_HPP
#define ACCURATE_MATH_HPP

#include <array>
#include <cmath>

template <typename T>
std::array<T, 2> two_prod(T a, T b) {
    T prod = a * b;
    T err = std::fma(a, b, -prod);
    return {prod, err};
}

#endif