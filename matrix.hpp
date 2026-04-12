#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <algorithm>

class Matrix {
    public:
        int rows, cols;
        Matrix(int r, int c): 
            rows(r), cols(c), data(rows * cols, 0.0f) {}
        float* operator[](size_t row) {
            return &data[row*cols]; 
        }
        const float* operator[](size_t row) const {
            return &data[row*cols];
        }
        const Matrix operator*(const Matrix &m) const;
        Matrix operator*(float scalar) const; 
        Matrix operator-(const Matrix &m) const; 
        Matrix operator+(const Matrix &m) const; 
    private:
        // row - column design 
        std::vector<float> data;
};

inline const Matrix Matrix::operator*(const Matrix &m) const {
    if (cols != m.rows) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    Matrix out(rows, m.cols);
    for(size_t row = 0; row < out.rows; row++) {
        const float* row_a = (*this)[row];
        for(size_t col = 0; col < out.cols; col++) {
            float sum = 0.0f;
            for(size_t i = 0; i < cols; i++) {
                 sum += (*this)[row][i] * m[i][col];
            }
            out[row][col] += sum;
        }
    }
    return out;
}

inline Matrix Matrix::operator*(float scalar) const {
    Matrix out(rows, cols);
    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            out[i][j] = (*this)[i][j] * scalar;
        }
    }
    return out;
}

inline Matrix Matrix::operator-(const Matrix &m) const {
    if(cols != m.cols || rows != m.rows) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    Matrix out(rows, cols);
    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            out[i][j] = (*this)[i][j] - m[i][j];
        }
    }
    return out;
}

inline Matrix Matrix::operator+(const Matrix &m) const {
    if(cols != m.cols || rows != m.rows) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    Matrix out(rows, cols);
    for(size_t i = 0; i < out.rows; i++) {
        for(size_t j = 0; j < out.cols; j++) {
            out[i][j] = (*this)[i][j] + m[i][j];
        }
    }
    return out;
}

inline Matrix transpose(const Matrix &m) {
    Matrix out(m.cols, m.rows);
    for(size_t i = 0; i < m.rows; i++) {
        for(size_t j = 0; j < m.cols; j++) {
            out[j][i] = m[i][j];
        }
    }
    return out;
}

#endif
