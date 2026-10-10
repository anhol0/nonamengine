#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <span>
#include <functional>

class Tensor {
public:
    Tensor(std::vector<uint64_t> shape);

    // Element-wise operations, with broadcasting where applicable
    Tensor operator+(const Tensor& rhs) const;
    Tensor operator-(const Tensor& rhs) const;
    Tensor operator*(const Tensor& rhs) const;
    Tensor operator/(const Tensor& rhs) const;

    // Scalar operations
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // Named structural operations
    Tensor matmul(const Tensor& rhs) const;
    Tensor transposed() const;

    void fill(float value);

    template<typename Generator>

    requires std::invocable<Generator> &&
             std::convertible_to<
                std::invoke_result_t<Generator>,
                float>
    void fill_with(Generator&& generator) {
        for(float& value: data_) {
            value = static_cast<float>(
                std::invoke(generator)
            );
        }
    }

    uint64_t extent(uint64_t axis) const;
    uint64_t rows() const;
    uint64_t cols() const;

    template<std::integral Index>
    static std::size_t checked_index(Index index) {
        if constexpr (std::signed_integral<Index>) {
            if (index < 0) {
                throw std::out_of_range(
                    "Tensor index cannot be negative"
                );
            }
        }

        return static_cast<std::size_t>(index);
    }

    float& at(std::span<const uint64_t> indices);
    const float& at(std::span<const uint64_t> indices) const;

    template<std::integral... Indices>
    float& at(Indices... indices) {
        const std::array<uint64_t, sizeof...(Indices)> values {
            checked_index(indices)...
        };
        return at(std::span<const uint64_t>(values));
    }

    template<std::integral... Indices>
    const float& at(Indices... indices) const {
        const std::array<uint64_t, sizeof...(Indices)> values {
            checked_index(indices)...
        };
        return at(std::span<const uint64_t>(values));
    }

    float& operator()(std::size_t row, std::size_t column);
    const float& operator()(std::size_t row, std::size_t column) const;

    template<std::integral... Indices>
    float& operator()(Indices... indices) {
        return at(indices...);
    }

    template<std::integral... Indices>
    const float& operator()(Indices... indices) const {
        return at(indices...);
    }

    uint64_t rank() const noexcept;
    std::span<float> data() noexcept {
        return data_;
    }
    std::span<const float> data() const noexcept {
        return data_;
    }
    std::span<const std::size_t> shape() const noexcept {
        return shape_;
    }
    std::span<const std::size_t> strides() const noexcept {
        return strides_;
    }

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<float> data_;
};

namespace tensor_operations {
    Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor transpose(const Tensor& a);
    Tensor relu(const Tensor& a);
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor subtract(const Tensor& a, const Tensor& b);
    Tensor multiply(const Tensor& a, const Tensor& b);
    Tensor divide(const Tensor& a, const Tensor& b);
}
