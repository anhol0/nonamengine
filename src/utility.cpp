#include "include/utility.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>

void print_matrix(const Matrix &m) {
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            std::cout << m[i][j] << ", ";
        }
        std::cout << "\n";
    }
}

float randf() {
    srand(time(NULL));
    return (float)rand() / RAND_MAX;
}
