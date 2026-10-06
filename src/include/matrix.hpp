#pragma once

#include <vector>

class Matrix {
    public:
        size_t rows, cols;
        Matrix() {}
        Matrix(size_t r, size_t c):
            rows(r), cols(c), data(rows * cols, 0.0f) {}
        float* operator[](size_t row) {
            return &data[row*cols];
        }
        const float* operator[](size_t row) const {
            return &data[row*cols];
        }
        const Matrix operator*(const Matrix &m) const;
        Matrix operator*(float scalar) const;
        Matrix operator/(float scalar) const;
        Matrix operator-(const Matrix &m) const;
        Matrix operator+(const Matrix &m) const;
    private:
        // row - column design
        std::vector<float> data;
};

Matrix relu(const Matrix &m);
Matrix transpose(const Matrix &m);
Matrix broadcast_cols(const Matrix &m, int new_cols);
Matrix broadcast_rows(const Matrix &m, int new_rows);
