#include "test_support.hpp"
#include "../src/exceptions.hpp"
#include <array>
#include <limits>

namespace {

using BinaryOperation = Tensor (*)(const Tensor&, const Tensor&);
const std::array<BinaryOperation, 8> binary_operations {
    [](const Tensor& a, const Tensor& b) { return a + b; },
    [](const Tensor& a, const Tensor& b) { return a - b; },
    [](const Tensor& a, const Tensor& b) { return a * b; },
    [](const Tensor& a, const Tensor& b) { return a / b; },
    tensor_operations::add,
    tensor_operations::subtract,
    tensor_operations::multiply,
    tensor_operations::divide,
};

void check_equal_shape_arithmetic(const std::vector<uint64_t>& shape) {
    const auto a = tensor(shape, {-6, 0, 1.5f, 8, -3, 10});
    const auto b = tensor(shape, {-2, 4, 0.5f, -4, 2, 5});
    const auto original_a = a;
    const auto original_b = b;
    const std::array<Tensor, 4> expected {
        tensor(shape, {-8, 4, 2, 4, -1, 15}),
        tensor(shape, {-4, -4, 1, 12, -5, 5}),
        tensor(shape, {12, 0, 0.75f, -32, -6, 50}),
        tensor(shape, {3, 0, 3, -2, -1.5f, 2}),
    };
    for(std::size_t i = 0; i < binary_operations.size(); ++i) {
        auto result = binary_operations[i](a, b);
        equal(result, expected[i % expected.size()]);
        result.data()[0] = 100;
        equal(a, original_a);
        equal(b, original_b);
    }
    // A tensor may appear on both sides without its storage being modified.
    equal(a + a, tensor(shape, {-12, 0, 3, 16, -6, 20}));
    equal(a - a, tensor(shape, {0, 0, 0, 0, 0, 0}));
    equal(b * b, tensor(shape, {4, 16, 0.25f, 16, 4, 25}));
    equal(b / b, tensor(shape, {1, 1, 1, 1, 1, 1}));
    equal(a, original_a);
    equal(b, original_b);
}

} // namespace

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

void equal_rank1() {
    check_equal_shape_arithmetic({6});
}

void equal_rank2() {
    check_equal_shape_arithmetic({2, 3});
}

void equal_rank3() {
    check_equal_shape_arithmetic({1, 2, 3});
    check_equal_shape_arithmetic({2, 1, 3});
    check_equal_shape_arithmetic({2, 3, 1});
}

void equal_rank4() {
    check_equal_shape_arithmetic({1, 2, 1, 3});
    check_equal_shape_arithmetic({2, 1, 3, 1});
    check_equal_shape_arithmetic({1, 1, 2, 3});
}

void scalar_and_singleton_arithmetic() {
    // Rank zero holds one element; singleton axes must not collapse to rank zero.
    for(const auto& shape : std::vector<std::vector<uint64_t>>{
            {}, {1}, {1, 1}, {1, 1, 1}, {1, 1, 1, 1}}) {
        const auto a = tensor(shape, {6});
        const auto b = tensor(shape, {2});
        const std::array<Tensor, 4> expected {
            tensor(shape, {8}), tensor(shape, {4}), tensor(shape, {12}), tensor(shape, {3})
        };
        for(std::size_t i = 0; i < binary_operations.size(); ++i) {
            auto result = binary_operations[i](a, b);
            equal(result, expected[i % expected.size()]);
            result.data()[0] = -100;
            equal(a, tensor(shape, {6}));
            equal(b, tensor(shape, {2}));
        }
    }
}

void empty_arithmetic() {
    for(const auto& shape : std::vector<std::vector<uint64_t>>{
            {0}, {0, 3}, {2, 0}, {0, 2, 3}, {2, 0, 3}, {2, 3, 0}, {1, 2, 0, 3}, {2, 0, 0}}) {
        const Tensor a(shape), b(shape);
        for(const auto operation : binary_operations) {
            const auto result = operation(a, b);
            equal(result, a);
            check(result.data().empty(), "Empty operands must produce no data");
        }
    }
}

void arithmetic_tail() {
    // An odd count above common SIMD widths exercises bulk-loop remainders.
    for(const auto& shape : std::vector<std::vector<uint64_t>>{{17}, {1, 17, 1}}) {
        Tensor a(shape), b(shape);
        for(std::size_t i = 0; i < a.data().size(); ++i) {
            a.data()[i] = static_cast<float>(static_cast<int>(i) - 8) * 0.25f;
            b.data()[i] = 1.0f + static_cast<float>(i % 3) * 0.5f;
        }
        for(std::size_t operation = 0; operation < binary_operations.size(); ++operation) {
            const auto result = binary_operations[operation](a, b);
            check(std::ranges::equal(result.shape(), a.shape()), "Arithmetic must preserve the full shape");
            check(result.data().size() == a.data().size(), "Arithmetic must preserve the element count");
            for(std::size_t i = 0; i < a.data().size(); ++i) {
                const float lhs = a.data()[i], rhs = b.data()[i];
                const std::array<float, 4> expected {lhs + rhs, lhs - rhs, lhs * rhs, lhs / rhs};
                near(result.data()[i], expected[operation % expected.size()]);
            }
        }
    }
}

void arithmetic_shape_guards() {
    for(const auto operation : binary_operations) {
        // Equal element counts alone do not imply equal or compatible shapes.
        throws<std::invalid_argument>([&] { (void)operation(Tensor({2, 3}), Tensor({3, 2})); });
        throws<std::invalid_argument>([&] { (void)operation(Tensor({0, 2}), Tensor({0, 3})); });
        throws<EqualRankException>([&] { (void)operation(Tensor({6}), Tensor({2, 3})); });
        throws<EqualRankException>([&] { (void)operation(Tensor({0}), Tensor({0, 2})); });
        throws<EqualRankException>([&] { (void)operation(tensor({}, {2}), tensor({1}, {3})); });
    }
    const auto a = tensor({1, 3}, {2, 4, 8});
    const auto b = tensor({3, 1}, {1, 2, 4});
    const std::array<Tensor, 4> expected {
        tensor({3, 3}, {3, 5, 9, 4, 6, 10, 6, 8, 12}),
        tensor({3, 3}, {1, 3, 7, 0, 2, 6, -2, 0, 4}),
        tensor({3, 3}, {2, 4, 8, 4, 8, 16, 8, 16, 32}),
        tensor({3, 3}, {2, 4, 8, 1, 2, 4, 0.5f, 1, 2}),
    };
    for(std::size_t i = 0; i < binary_operations.size(); ++i) {
        equal(binary_operations[i](a, b), expected[i % expected.size()]);
    }
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
        else if(name == "equal_rank1") equal_rank1();
        else if(name == "equal_rank2") equal_rank2();
        else if(name == "equal_rank3") equal_rank3();
        else if(name == "equal_rank4") equal_rank4();
        else if(name == "scalar_and_singleton") scalar_and_singleton_arithmetic();
        else if(name == "empty_arithmetic") empty_arithmetic();
        else if(name == "arithmetic_tail") arithmetic_tail();
        else if(name == "shape_guards") arithmetic_shape_guards();
        else if(name == "structural") structural();
        else throw std::runtime_error("Unknown case");
    });
}
