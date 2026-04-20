#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "matrix.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>

inline void print_matrix(const Matrix &m) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            std::cout << m[i][j] << ", ";
        }
        std::cout << "\n";
    }
}

inline float randf() {
    srand(time(NULL));
    return (float)rand() / RAND_MAX;
}

#endif
