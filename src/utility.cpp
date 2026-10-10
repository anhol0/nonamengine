#include "include/utility.hpp"
#include "exceptions.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>
#include <random>

void print_matrix(const Tensor &m) {
    if(m.rank() != 2)
        throw RequiredRankException(2);

    for(size_t i = 0; i < m.rows(); i++) {
        for(size_t j = 0; j < m.cols(); j++) {
            std::cout << m.at(i, j) << ", ";
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
