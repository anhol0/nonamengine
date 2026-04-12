#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "matrix.hpp"
#include <iostream>

inline void print_matrix(const Matrix &m) {
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            std::cout << m[i][j] << ", ";
        }
        std::cout << "\n";
    }
}

inline float randf() {
    return (float)rand() / RAND_MAX;
}

#endif
