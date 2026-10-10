#include "include/matrix.hpp"
#include "exceptions.hpp"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

// Tensors

namespace {
    void require_equal_rank(const Tensor& a, const Tensor& b) {
        if(a.rank() != b.rank()) throw EqualRankException();
    }

    template<uint64_t N, std::derived_from<Tensor>... Tensors>
    void require_rank(const Tensors&... tensors) {
        if(!((tensors.rank() == N) && ...))
            throw RequiredRankException(N);
    }

    std::vector<std::size_t> broadcast_shape(const Tensor& a,
                                             const Tensor& b) {
        require_equal_rank(a, b);

        std::vector<std::size_t> result(a.rank());
        for(std::size_t i = 0; i < result.size(); i++) {
            const std::size_t a_dimension = a.shape()[i];
            const std::size_t b_dimension = b.shape()[i];
            if(a_dimension != b_dimension &&
               a_dimension != 1 && b_dimension != 1) {
                throw std::invalid_argument(
                    "Tensor dimensions are not broadcast-compatible");
            }
            result[i] = std::max(a_dimension, b_dimension);
        }
        return result;
    }

    template<typename Operation>
    Tensor elementwise_binary(const Tensor& a, const Tensor& b,
                              Operation operation) {
        const std::vector<std::size_t> result_shape = broadcast_shape(a, b);
        Tensor result(result_shape);

        for(std::size_t flat_index = 0;
            flat_index < result.data().size(); flat_index++) {
            std::size_t remaining = flat_index;
            std::size_t a_offset = 0;
            std::size_t b_offset = 0;

            for(std::size_t dimension = 0;
                dimension < result_shape.size(); dimension++) {
                const std::size_t index =
                    remaining / result.strides()[dimension];
                remaining %= result.strides()[dimension];

                if(a.shape()[dimension] != 1) {
                    a_offset += index * a.strides()[dimension];
                }
                if(b.shape()[dimension] != 1) {
                    b_offset += index * b.strides()[dimension];
                }
            }

            result.data()[flat_index] =
                operation(a.data()[a_offset], b.data()[b_offset]);
        }
        return result;
    }

    template<typename Operation>
    Tensor elementwise_scalar(const Tensor& tensor, float scalar,
                              Operation operation) {
        Tensor result(std::vector<std::size_t>(tensor.shape().begin(),
                                               tensor.shape().end()));
        std::transform(tensor.data().begin(), tensor.data().end(),
                       result.data().begin(),
                       [scalar, operation](float value) {
                           return operation(value, scalar);
                       });
        return result;
    }
}

Tensor tensor_operations::matmul(const Tensor &a, const Tensor &b) {
    require_rank<2>(a, b);
    if(a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument(
            "Tensor dimensions are incompatible for matrix multiplication");
    }

    Tensor result({a.shape()[0], b.shape()[1]});
    for(std::size_t row = 0; row < result.shape()[0]; row++) {
        for(std::size_t column = 0; column < result.shape()[1]; column++) {
            float sum = 0.0f;
            for(std::size_t i = 0; i < a.shape()[1]; i++) {
                sum += a(row, i) * b(i, column);
            }
            result(row, column) = sum;
        }
    }
    return result;
}

Tensor tensor_operations::transpose(const Tensor& tensor) {
    require_rank<2>(tensor);

    Tensor result({tensor.shape()[1], tensor.shape()[0]});
    for(std::size_t row = 0; row < tensor.shape()[0]; row++) {
        for(std::size_t column = 0; column < tensor.shape()[1]; column++) {
            result(column, row) = tensor(row, column);
        }
    }
    return result;
}

Tensor tensor_operations::relu(const Tensor& tensor) {
    return elementwise_scalar(tensor, 0.0f,
        [](float value, float) { return std::max(0.0f, value); });
}

Tensor tensor_operations::add(const Tensor& a, const Tensor& b) {
    return elementwise_binary(a, b,
        [](float lhs, float rhs) { return lhs + rhs; });
}

Tensor tensor_operations::subtract(const Tensor& a, const Tensor& b) {
    return elementwise_binary(a, b,
        [](float lhs, float rhs) { return lhs - rhs; });
}

Tensor tensor_operations::multiply(const Tensor& a, const Tensor& b) {
    return elementwise_binary(a, b,
        [](float lhs, float rhs) { return lhs * rhs; });
}

Tensor tensor_operations::divide(const Tensor& a, const Tensor& b) {
    return elementwise_binary(a, b,
        [](float lhs, float rhs) { return lhs / rhs; });
}

Tensor Tensor::operator+(const Tensor& rhs) const {
    return tensor_operations::add(*this, rhs);
}

Tensor Tensor::operator-(const Tensor& rhs) const {
    return tensor_operations::subtract(*this, rhs);
}

Tensor Tensor::operator*(const Tensor& rhs) const {
    return tensor_operations::multiply(*this, rhs);
}

Tensor Tensor::operator/(const Tensor& rhs) const {
    return tensor_operations::divide(*this, rhs);
}

Tensor Tensor::operator+(float scalar) const {
    return elementwise_scalar(*this, scalar,
        [](float value, float operand) { return value + operand; });
}

Tensor Tensor::operator-(float scalar) const {
    return elementwise_scalar(*this, scalar,
        [](float value, float operand) { return value - operand; });
}

Tensor Tensor::operator*(float scalar) const {
    return elementwise_scalar(*this, scalar,
        [](float value, float operand) { return value * operand; });
}

Tensor Tensor::operator/(float scalar) const {
    return elementwise_scalar(*this, scalar,
        [](float value, float operand) { return value / operand; });
}

Tensor Tensor::matmul(const Tensor& rhs) const {
    return tensor_operations::matmul(*this, rhs);
}

Tensor Tensor::transposed() const {
    return tensor_operations::transpose(*this);
}

float& Tensor::operator()(std::size_t row, std::size_t column) {
    require_rank<2>(*this);
    if(row >= shape_[0] || column >= shape_[1]) {
        throw std::out_of_range("Tensor index is out of range");
    }
    return data_[row * strides_[0] + column * strides_[1]];
}

const float& Tensor::operator()(std::size_t row,
                                std::size_t column) const {
    require_rank<2>(*this);
    if(row >= shape_[0] || column >= shape_[1]) {
        throw std::out_of_range("Tensor index is out of range");
    }
    return data_[row * strides_[0] + column * strides_[1]];
}

uint64_t Tensor::extent(uint64_t axis) const {
    if(axis >= shape_.size()) {
        throw std::out_of_range("Tensor axis out of range");
    }
    return shape_[axis];
}

uint64_t Tensor::rows() const {
    require_rank<2>(*this);
    return extent(0);
}

uint64_t Tensor::cols() const {
    require_rank<2>(*this);
    return extent(1);
}

float& Tensor::at(std::span<const uint64_t> indices) {
    if(indices.size() != rank()) {
        throw std::invalid_argument(
            "Number of indices must equal tensor rank"
        );
    }

    uint64_t offset = 0;
    for(uint64_t axis = 0; axis < rank(); axis++) {
        if (indices[axis] >= shape_[axis]) {
            throw std::out_of_range(
                "Tensor index is out of range"
            );
        }
        offset += indices[axis] * strides_[axis];
    }

    return data_[offset];
}

const float& Tensor::at(std::span<const uint64_t> indices) const {
    if(indices.size() != rank()) {
        throw std::invalid_argument(
            "Number of indices must equal tensor rank"
        );
    }

    uint64_t offset = 0;
    for(uint64_t axis = 0; axis < rank(); axis++) {
        if (indices[axis] >= shape_[axis]) {
            throw std::out_of_range(
                "Tensor index is out of range"
            );
        }
        offset += indices[axis] * strides_[axis];
    }

    return data_[offset];
}

void Tensor::fill(float value) {
    for(float& val : data_) {
        val = value;
    }
}



Tensor::Tensor(std::vector<uint64_t> shape) :
    shape_(std::move(shape)),
    strides_(shape_.size())
{
    std::size_t element_count = 1;

    for(std::size_t i = shape_.size(); i-- > 0;) {
        strides_[i] = element_count;
        if(shape_[i] != 0 &&
           element_count > std::numeric_limits<std::size_t>::max() / shape_[i]) {
            throw std::length_error("Tensor element count is too large");
        }
        element_count *= shape_[i];
    }

    data_.resize(element_count, 0.0f);
}

uint64_t Tensor::rank() const noexcept {
    return shape_.size();
}
