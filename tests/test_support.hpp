#pragma once

#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

inline void check(bool condition, const std::string& message) {
    if(!condition) throw std::runtime_error(message);
}

inline void near(float actual, float expected, float tolerance = 1e-5f) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          "Expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

template<class Exception, class Function>
void throws(Function function) {
    try { function(); }
    catch(const Exception&) { return; }
    throw std::runtime_error("Expected exception was not thrown");
}

inline Tensor tensor(std::vector<uint64_t> shape, std::initializer_list<float> values) {
    Tensor result(std::move(shape));
    check(result.data().size() == values.size(), "Invalid test fixture");
    std::copy(values.begin(), values.end(), result.data().begin());
    return result;
}

inline void values(const Tensor& actual, std::initializer_list<float> expected) {
    check(actual.data().size() == expected.size(), "Unexpected tensor size");
    std::size_t i = 0;
    for(float value : expected) near(actual.data()[i++], value);
}

inline void equal(const Tensor& actual, const Tensor& expected, float tolerance = 1e-5f) {
    check(std::ranges::equal(actual.shape(), expected.shape()), "Unexpected tensor shape");
    check(actual.data().size() == expected.data().size(), "Unexpected tensor size");
    for(std::size_t i = 0; i < actual.data().size(); ++i)
        near(actual.data()[i], expected.data()[i], tolerance);
}

// Own only the directory successfully created here, including on test failures.
struct TestDirectory {
    std::filesystem::path path;
    TestDirectory() {
        std::random_device random;
        for(int attempt = 0; attempt < 100; ++attempt) {
            auto candidate = std::filesystem::temp_directory_path() /
                ("nonamengine-test-" + std::to_string(random()));
            if(std::filesystem::create_directory(candidate)) {
                path = std::move(candidate);
                return;
            }
        }
        throw std::runtime_error("Could not create test directory");
    }
    ~TestDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    TestDirectory(const TestDirectory&) = delete;
    TestDirectory& operator=(const TestDirectory&) = delete;
};

template<class Function>
int run_test(Function function) {
    try { function(); return 0; }
    catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
