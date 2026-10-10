#pragma once

#include <cstdint>
#include <exception>
#include <string>

class EqualRankException : public std::exception {
public:
    const char* what() const noexcept override {
        return "Two tensors have to have equal rank for this operation";
    }
};

class RequiredRankException : public std::exception {
    uint64_t rank_;
    std::string str;
public:
    explicit RequiredRankException(uint64_t rank) :
        rank_(rank),
        str("Required tensor rank is " + std::to_string(rank_))
    {}

    const char* what() const noexcept override {
        return str.c_str();
    }
};
