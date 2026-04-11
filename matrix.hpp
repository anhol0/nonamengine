#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <algorithm>

class Matrix {
    public:
        int rows, cols;
        Matrix(int r, int c): 
            rows(r), cols(c), data(rows, std::vector<float>(cols, 0.0f)) {}
        std::vector<float>& operator[](size_t row) {
            return data[row]; 
        }
        const std::vector<float>& operator[](size_t row) const {
            return data[row];
        }
        const Matrix operator*(const Matrix &m) const {
            if (cols != m.rows) {
                throw std::invalid_argument("Invalid matrix dimensions");
            }
            Matrix out(rows, m.cols);
            for(int row = 0; row < out.rows; row++) {
                for(int col = 0; col < out.cols; col++) {
                    for(int i = 0; i < cols; i++) {
                        out[row][col] += data[row][i] * m[i][col];
                    }
                }
            }
            return out;
        }
        Matrix operator*(float scalar) const {
            Matrix out(rows, cols);
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    out[i][j] = data[i][j] * scalar;
                }
            }
            return out;
        }
        Matrix operator-(const Matrix &m) const {
            Matrix out(rows, cols);
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    out[i][j] = data[i][j] - m[i][j];
                }
            }
            return out;
        }
        const Matrix operator+(const Matrix &m) const {
            if(cols != m.cols || rows != m.rows) {
                throw std::invalid_argument("Invalid matrix dimensions");
            }
            Matrix out(rows, cols);
            for(int i = 0; i < out.rows; i++) {
                for(int j = 0; j < out.cols; j++) {
                    out[i][j] = data[i][j] + m[i][j];
                }
            }
            return out;
        }
    private:
        // row - column design 
        std::vector<std::vector<float>> data;
};

inline Matrix subtract(const Matrix &m1, const Matrix &m2) {
    if(m1.cols != m2.cols || m1.rows != m2.rows) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    Matrix out(m1.rows, m1.cols);
    for(int i = 0; i < out.rows; i++) {
        for(int j = 0; j < out.cols; j++) {
            out[i][j] = m1[i][j] - m2[i][j];
        }
    }
    return out;
}

inline Matrix relu(const Matrix &m) {
    Matrix out(m.rows, m.cols);

    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            out[i][j] = std::max(0.0f, m[i][j]);
        }
    }
    return out;
}

inline Matrix transpose(const Matrix &m) {
    Matrix out(m.cols, m.rows);
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            out[j][i] = m[i][j];
        }
    }
    return out;
}

#endif
