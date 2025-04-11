#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include <mpfr.h>

#include "accurate_math.hpp"
#include "kobbelt.hpp"

using namespace std;
using namespace std::chrono;

template <typename T>
void generate_vector(T* vec, size_t size, T min = -1e6, T max = 1e6) {
    random_device rd;
    mt19937_64 gen(rd());
    uniform_real_distribution<T> dist(min, max);
    
    for (size_t i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }
}

template <typename T>
T naive_dot(const T* a, const T* b, size_t size) {
    T result = 0;
    for (size_t i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

template <typename T>
T kahan_dot(const T* a, const T* b, size_t size) {
    T sum = 0, err = 0;
    for (size_t i = 0; i < size; ++i) {
        T product = a[i] * b[i];
        T adjusted = product - err;
        T new_sum = sum + adjusted;
        err = (new_sum - sum) - adjusted;
        sum = new_sum;
    }
    return sum;
}

long double exact_dot(const double* a, const double* b, size_t size) {
    mpfr_t sum, product, fa, fb;
    const int PRECISION = 256;
    
    mpfr_inits2(PRECISION, sum, product, fa, fb, nullptr);
    mpfr_set_d(sum, 0.0, MPFR_RNDN);
    
    for (size_t i = 0; i < size; ++i) {
        mpfr_set_d(fa, a[i], MPFR_RNDN);
        mpfr_set_d(fb, b[i], MPFR_RNDN);
        mpfr_mul(product, fa, fb, MPFR_RNDN);
        mpfr_add(sum, sum, product, MPFR_RNDN);
    }
    
    long double result = mpfr_get_ld(sum, MPFR_RNDN);
    mpfr_clears(sum, product, fa, fb, nullptr);
    return result;
}

struct BenchmarkResult {
    double time;
    long double error;
    int bits_loss;
};

template <typename Func>
BenchmarkResult benchmark(Func&& func, const double* a, const double* b, 
                          size_t size, long double exact) {
    auto start = high_resolution_clock::now();
    auto result = func(a, b, size);
    auto end = high_resolution_clock::now();
    
    long double error = fabsl(exact - result);
    int bits_loss = 0;
    if (error > 0 && exact != 0.0) {
        bits_loss = std::max(0, static_cast<int>(std::ceil(-log2l(error / fabsl(exact)))));
    }

    
    return {
        duration<double>(end - start).count(),
        error,
        bits_loss
    };
}

template <typename T>
void print_vector(const std::vector<T>& vec, const std::string& name) {
    cout << name << " = [";
    size_t n = vec.size();
    if(n <= 20) {
        for (size_t i = 0; i < n; ++i) {
            cout << vec[i];
            if (i != n - 1)
                cout << ", ";
        }
    } else {
        for (size_t i = 0; i < 10; ++i)
            cout << vec[i] << ", ";
        cout << "... , ";
        for (size_t i = n - 10; i < n; ++i) {
            cout << vec[i];
            if (i != n - 1)
                cout << ", ";
        }
    }
    cout << "]\n";
}

void write_results_to_csv(const std::string& filename,
    const std::vector<BenchmarkResult>& resultsNaive,
    const std::vector<BenchmarkResult>& resultsKahan,
    const std::vector<BenchmarkResult>& resultsKobbelt)
{
    ofstream ofs(filename);
    ofs << "Iteration,Method,Time,Error,Bits_lost\n";
    
    auto write_vector = [&](const std::string& method, const std::vector<BenchmarkResult>& results) {
        for (size_t i = 0; i < results.size(); ++i) {
            ofs << i << "," << method << "," 
                << results[i].time << ","
                << results[i].error << ","
                << results[i].bits_loss << "\n";
        }
    };
    
    write_vector("Naive", resultsNaive);
    write_vector("Kahan", resultsKahan);
    write_vector("Kobbelt", resultsKobbelt);
    
    ofs.close();
}

int main(int argc, char** argv) {
    const size_t VECTOR_SIZE = 1024;
    const int ITERATIONS = 1000;
    
    vector<double> a(VECTOR_SIZE);
    vector<double> b(VECTOR_SIZE);
    
    generate_vector(a.data(), VECTOR_SIZE);
    generate_vector(b.data(), VECTOR_SIZE);
    
    cout << "Векторы, участвующие в умножении:\n";
    print_vector(a, "Вектор a");
    print_vector(b, "Вектор b");
    cout << "\n";
    
    auto exact = exact_dot(a.data(), b.data(), VECTOR_SIZE);
    
    vector<BenchmarkResult> resultsNaive;
    vector<BenchmarkResult> resultsKahan;
    vector<BenchmarkResult> resultsKobbelt;
    
    for (int i = 0; i < ITERATIONS; ++i) {
        resultsNaive.push_back(benchmark(naive_dot<double>, a.data(), b.data(), VECTOR_SIZE, exact));
    }

    for (int i = 0; i < ITERATIONS; ++i) {
        resultsKahan.push_back(benchmark(kahan_dot<double>, a.data(), b.data(), VECTOR_SIZE, exact));
    }

    for (int i = 0; i < ITERATIONS; ++i) {
        resultsKobbelt.push_back(benchmark(kobbelt_dot_product<double>, a.data(), b.data(), VECTOR_SIZE, exact));
    }
    
    auto compute_average = [&](const std::vector<BenchmarkResult>& vec) -> BenchmarkResult {
        BenchmarkResult avg {0, 0, 0};
        for (const auto& res : vec) {
            avg.time += res.time;
            avg.error += res.error;
            avg.bits_loss = max(avg.bits_loss, res.bits_loss);
        }
        avg.time /= vec.size();
        avg.error /= vec.size();
        return avg;
    };
    
    auto avgNaive   = compute_average(resultsNaive);
    auto avgKahan   = compute_average(resultsKahan);
    auto avgKobbelt = compute_average(resultsKobbelt);
    
    cout << "Метод: Naive"
         << "\nСреднее время: " << avgNaive.time << " с"
         << "\nСредняя ошибка: " << avgNaive.error
         << "\nМаксимальное потеряно бит: " << avgNaive.bits_loss << "\n\n";

    cout << "Метод: Kahan"
         << "\nСреднее время: " << avgKahan.time << " с"
         << "\nСредняя ошибка: " << avgKahan.error
         << "\nМаксимальное потеряно бит: " << avgKahan.bits_loss << "\n\n";

    cout << "Метод: Kobbelt"
         << "\nСреднее время: " << avgKobbelt.time << " с"
         << "\nСредняя ошибка: " << avgKobbelt.error
         << "\nМаксимальное потеряно бит: " << avgKobbelt.bits_loss << "\n\n";
    
    // Экспорт результатов в CSV-файл для последующей визуализации
    write_results_to_csv("benchmark_results.csv", resultsNaive, resultsKahan, resultsKobbelt);
    
    ofstream fout("scale_results.csv");
    fout << "Size,Method,Error,Bits_lost\n";

    vector<size_t> sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    const int REPEATS_PER_SIZE = 20;

    for (size_t size : sizes) {
        vector<double> a(size);
        vector<double> b(size);

        auto run = [&](auto func, const string& name) {
            long double total_error = 0;
            int max_bits_lost = 0;

            for (int i = 0; i < REPEATS_PER_SIZE; ++i) {
                generate_vector(a.data(), size);
                generate_vector(b.data(), size);
                auto exact = exact_dot(a.data(), b.data(), size);

                auto res = benchmark(func, a.data(), b.data(), size, exact);
                total_error += res.error;
                max_bits_lost = max(max_bits_lost, res.bits_loss);
            }

            long double avg_error = total_error / REPEATS_PER_SIZE;
            fout << size << "," << name << "," << avg_error << "," << max_bits_lost << "\n";
        };

        run(naive_dot<double>, "Naive");
        run(kahan_dot<double>, "Kahan");
        run(kobbelt_dot_product<double>, "Kobbelt");
    }

    return 0;
}
