#include "include/utility.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>
#include <random>

void print_matrix(const Matrix &m) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            std::cout << m[i][j] << ", ";
        }
        std::cout << "\n";
    }
}

float randf() {
    std::random_device rand;
    std::mt19937 gen(rand());
    std::uniform_real_distribution<> res(0.f, 1.f);
    return res(gen);
}
