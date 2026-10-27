#include "test_support.hpp"
#include "../src/exceptions.hpp"
#include <limits>

void construction() {
    Tensor t({2, 3, 4});
    check(t.rank() == 3 && t.extent(1) == 3, "Rank/extents");
    check(std::ranges::equal(t.strides(), std::vector<std::size_t>{12, 4, 1}), "Strides");
    for(float value : t.data()) near(value, 0);
    t.at(1, 2, 3) = 7;
    near(t.data()[23], 7);
    const Tensor& c = t;
    near(c(1, 2, 3), 7);
    Tensor copy = t;
    copy.at(1, 2, 3) = 8;
    near(t.at(1, 2, 3), 7);
    Tensor moved = std::move(copy);
    near(moved.at(1, 2, 3), 8);
    Tensor scalar({});
    scalar.at() = 3;
    near(std::as_const(scalar).at(), 3);
    Tensor empty({2, 0, 3});
    check(empty.data().empty(), "Empty tensor storage");
    t.fill(2);
    int calls = 0;
    t.fill_with([&] { return ++calls; });
    near(t.data().front(), 1);
    near(t.data().back(), 24);
    throws<std::length_error>([] { Tensor huge({std::numeric_limits<uint64_t>::max(), 2}); });
    throws<std::out_of_range>([&] { (void)t.extent(3); });
    throws<RequiredRankException>([&] { (void)t.rows(); });
    throws<RequiredRankException>([&] { (void)t.cols(); });
    throws<std::out_of_range>([&] { (void)t.at(-1, 0, 0); });
    throws<std::out_of_range>([&] { (void)c.at(2, 0, 0); });
    throws<std::out_of_range>([&] { (void)t.at(0, 3, 0); });
    throws<std::invalid_argument>([&] { (void)t.at(0); });
    throws<std::invalid_argument>([&] { (void)c.at(0); });
    throws<std::out_of_range>([&] { (void)empty.at(0, 0, 0); });
    Tensor matrix({2, 3});
    std::size_t row = 1, col = 2;
    matrix(row, col) = 5;
    near(std::as_const(matrix)(row, col), 5);
    check(matrix.rows() == 2 && matrix.cols() == 3, "Rows/columns");
    throws<std::out_of_range>([&] { (void)matrix(std::size_t(2), col); });
    throws<std::out_of_range>([&] { (void)std::as_const(matrix)(row, std::size_t(3)); });
    throws<RequiredRankException>([&] { (void)t(row, col); });
    throws<RequiredRankException>([&] { (void)c(row, col); });
    throws<std::out_of_range>([&] { (void)std::as_const(matrix).at(-1, 0); });
}

void arithmetic() {
    auto a = tensor({2, 2}, {2, 4, 6, 8});
    auto b = tensor({2, 2}, {1, 2, 3, 4});
    values(a + b, {3, 6, 9, 12});
    values(a - b, {1, 2, 3, 4});
    values(a * b, {2, 8, 18, 32});
    values(a / b, {2, 2, 2, 2});
    values(a + 2, {4, 6, 8, 10});
    values(a - 2, {0, 2, 4, 6});
    values(a * 2, {4, 8, 12, 16});
    values(a / 2, {1, 2, 3, 4});
    values(a + tensor({2, 1}, {10, 20}), {12, 14, 26, 28});
    values(tensor({1, 2}, {10, 20}) - a, {8, 16, 4, 12});
    values(a * tensor({1, 1}, {2}), {4, 8, 12, 16});
    values(a / tensor({2, 1}, {2, 4}), {1, 2, 1.5f, 2});
    auto higher = tensor({2, 1, 2}, {1, 2, 3, 4}) + tensor({1, 2, 1}, {10, 20});
    check(std::ranges::equal(higher.shape(), std::vector<std::size_t>{2, 2, 2}), "Broadcast shape");
    values(higher, {11, 12, 21, 22, 13, 14, 23, 24});
    values(tensor({}, {2}) + tensor({}, {3}), {5});
    check((Tensor({0, 2}) + Tensor({1, 2})).data().empty(), "Empty broadcast must stay empty");
    throws<EqualRankException>([&] { (void)(a + Tensor({2})); });
    throws<std::invalid_argument>([&] { (void)(a + Tensor({3, 2})); });
    values(a, {2, 4, 6, 8});
}

void structural() {
    auto a = tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = tensor({3, 2}, {7, 8, 9, 10, 11, 12});
    values(a.matmul(b), {58, 64, 139, 154});
    values(a.transposed(), {1, 4, 2, 5, 3, 6});
    equal(a.transposed().transposed(), a);
    equal(tensor_operations::matmul(a, b), a.matmul(b));
    equal(tensor_operations::transpose(a), a.transposed());
    values(Tensor({2, 0}).matmul(Tensor({0, 3})), {0, 0, 0, 0, 0, 0});
    check(Tensor({0, 2}).transposed().data().empty(), "Empty transpose");
    values(tensor_operations::relu(tensor({4}, {-2, 0, 1, 3})), {0, 0, 1, 3});
    throws<std::invalid_argument>([&] { (void)a.matmul(a); });
    throws<RequiredRankException>([&] { (void)a.matmul(Tensor({3})); });
    throws<RequiredRankException>([] { (void)Tensor({3}).transposed(); });
}

int main(int argc, char** argv) {
    return run_test([&] {
        check(argc == 2, "Expected case name");
        std::string name = argv[1];
        if(name == "construction") construction();
        else if(name == "arithmetic") arithmetic();
        else if(name == "structural") structural();
        else throw std::runtime_error("Unknown case");
    });
}
