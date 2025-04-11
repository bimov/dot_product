#ifndef KOBBELT_HPP
#define KOBBELT_HPP

#include <map>
#include <array>
#include "accurate_math.hpp"

template <typename T>
constexpr int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename T>
int compute_genus(T value) {
    int exp;
    std::frexp(value, &exp);
    return exp * 2 + (std::ilogb(value) & 1);
}

template <typename T>
void table_insert(std::map<int, T>& table, T value) {
    int genus = compute_genus(value);
    
    if (auto it = table.find(genus); it != table.end()) {
        value += it->second;
        table.erase(it);
        table_insert(table, value);
    } else {
        int paired_genus = genus ^ 1;
        if (auto paired_it = table.find(paired_genus); 
            paired_it != table.end() && sign(value) != sign(paired_it->second)) {
            value += paired_it->second;
            table.erase(paired_it);
            table_insert(table, value);
        } else {
            table.emplace(genus, value);
        }
    }
}

template <typename T>
T kobbelt_dot_product(const T* a, const T* b, size_t size) {
    std::map<int, T> table;
    
    for (size_t i = 0; i < size; ++i) {
        auto [prod, err] = two_prod(a[i], b[i]);
        table_insert(table, prod);
        table_insert(table, err);
    }
    
    T result = 0;
    for (const auto& [_, val] : table) {
        result += val;
    }
    return result;
}

#endif