#include "test_support.hpp"
#include "nn.hpp"
#include "model.hpp"
#include "utility.hpp"
#include "../src/exceptions.hpp"
#include <sstream>

void dense() {
    Dense layer(tensor({2, 2}, {1, 2, 3, 4}), tensor({2, 1}, {1, -1}));
    auto input = tensor({2, 2}, {1, 2, 3, 4});
    values(layer.forward(input), {8, 11, 14, 21});
    values(layer.backward(tensor({2, 2}, {1, 2, 3, 4})), {10, 14, 14, 20});
    values(*layer.grad_weights, {5, 11, 11, 25});
    values(*layer.grad_biases, {3, 7});
    layer.update(0.1f);
    values(layer.weights, {0.5f, 0.9f, 1.9f, 1.5f});
    values(layer.biases, {0.7f, -1.7f});
    auto parameters = layer.parameters();
    check(parameters.size() == 2 && parameters[0].first == "weights" &&
          &parameters[0].second.get() == &layer.weights, "Parameter references");
    check(layer.export_json()["input_size"] == 2, "Dense metadata");
    Dense initialized(3, 4);
    for(float value : initialized.weights.data())
        check(std::isfinite(value) && std::abs(value) <= std::sqrt(1.0f / 3), "Initialization range");
    for(float value : initialized.biases.data()) near(value, 0);
    throws<std::logic_error>([&] { initialized.update(0.1f); });
    throws<std::logic_error>([&] { (void)initialized.backward(Tensor({4, 1})); });
    throws<std::invalid_argument>([] { Dense invalid(0, 2); });
    throws<std::invalid_argument>([] { Dense invalid(2, 0); });
    throws<std::invalid_argument>([] { Dense invalid(Tensor({2}), Tensor({2, 1})); });
    throws<std::invalid_argument>([] { Dense invalid(Tensor({2, 2}), Tensor({1, 1})); });
    throws<std::invalid_argument>([] { Dense invalid(Tensor({2, 0}), Tensor({2, 1})); });
    throws<std::invalid_argument>([&] { (void)layer.forward(Tensor({3, 1})); });
    layer.forward(input);
    throws<std::invalid_argument>([&] { (void)layer.backward(Tensor({1, 2})); });
    throws<std::invalid_argument>([&] { (void)layer.backward(Tensor({2, 1})); });
    throws<std::invalid_argument>([&] { (void)layer.backward(Tensor({2})); });
}

void activations() {
    ReLU relu;
    Tanh tanh;
    throws<std::logic_error>([&] { (void)relu.backward(Tensor({1, 1})); });
    throws<std::logic_error>([&] { (void)tanh.backward(Tensor({1, 1})); });
    auto input = tensor({1, 3}, {-1, 0, 1});
    values(relu.forward(input), {0, 0, 1});
    values(relu.backward(tensor({1, 3}, {2, 3, 4})), {0, 0, 4});
    auto out = tanh.forward(input);
    auto grad = tanh.backward(tensor({1, 3}, {2, 3, 4}));
    for(std::size_t i = 0; i < 3; ++i) {
        near(out.data()[i], std::tanh(input.data()[i]));
        near(grad.data()[i], (2.0f + i) * (1 - out.data()[i] * out.data()[i]));
    }
    for(Layer* layer : std::vector<Layer*>{&relu, &tanh}) {
        throws<std::invalid_argument>([&] { (void)layer->backward(Tensor({1, 4})); });
        throws<std::invalid_argument>([&] { (void)layer->backward(Tensor({2, 3})); });
        layer->update(0.1f);
        check(layer->parameters().empty(), "Activation has no parameters");
    }
    check(relu.export_json()["type"] == "relu" && tanh.export_json()["type"] == "tanh", "Activation metadata");
}

void loss() {
    MSELoss mse;
    auto pred = tensor({2, 2}, {1, 2, 3, 4});
    auto target = tensor({2, 2}, {0, 2, 1, 6});
    near(mse.forward(pred, target), 2.25f);
    values(mse.backward(pred, target), {0.5f, 0, 1, -1});
    near(mse.forward(pred, pred), 0);
    throws<std::invalid_argument>([&] { (void)mse.forward(pred, Tensor({2, 1})); });
    throws<std::invalid_argument>([&] { (void)mse.backward(pred, Tensor({1, 2})); });
    throws<std::invalid_argument>([&] { (void)mse.forward(Tensor({0, 2}), Tensor({0, 2})); });
    throws<std::invalid_argument>([&] { (void)mse.backward(Tensor({0, 2}), Tensor({0, 2})); });
}

float objective(Layer& layer, const Tensor& input, const Tensor& upstream) {
    auto output = layer.forward(input);
    float sum = 0;
    for(std::size_t i = 0; i < output.data().size(); ++i)
        sum += output.data()[i] * upstream.data()[i];
    return sum;
}

void gradients() {
    Dense dense(tensor({2, 2}, {0.3f, -0.4f, 0.2f, 0.7f}), tensor({2, 1}, {0.1f, -0.2f}));
    ReLU relu;
    Tanh tanh;
    auto input = tensor({2, 2}, {-0.8f, 0.4f, 0.6f, -0.3f});
    auto upstream = tensor({2, 2}, {0.2f, -0.3f, 0.5f, 0.7f});
    constexpr float epsilon = 0.001f;
    for(Layer* layer : std::vector<Layer*>{&dense, &relu, &tanh}) {
        layer->forward(input);
        auto analytic = layer->backward(upstream);
        for(std::size_t i = 0; i < input.data().size(); ++i) {
            auto perturbed = input;
            perturbed.data()[i] += epsilon;
            float plus = objective(*layer, perturbed, upstream);
            perturbed.data()[i] -= 2 * epsilon;
            float minus = objective(*layer, perturbed, upstream);
            near(analytic.data()[i], (plus - minus) / (2 * epsilon), 0.0002f);
        }
    }
    dense.forward(input);
    dense.backward(upstream);
    for(auto pair : {std::pair{&dense.weights, &*dense.grad_weights},
                     std::pair{&dense.biases, &*dense.grad_biases}}) {
        for(std::size_t i = 0; i < pair.first->data().size(); ++i) {
            float original = pair.first->data()[i];
            pair.first->data()[i] = original + epsilon;
            float plus = objective(dense, input, upstream);
            pair.first->data()[i] = original - epsilon;
            float minus = objective(dense, input, upstream);
            pair.first->data()[i] = original;
            near(pair.second->data()[i], (plus - minus) / (2 * epsilon), 0.0002f);
        }
    }
    MSELoss mse;
    auto analytic = mse.backward(input, upstream);
    for(std::size_t i = 0; i < input.data().size(); ++i) {
        auto perturbed = input;
        perturbed.data()[i] += epsilon;
        float plus = mse.forward(perturbed, upstream);
        perturbed.data()[i] -= 2 * epsilon;
        float minus = mse.forward(perturbed, upstream);
        near(analytic.data()[i], (plus - minus) / (2 * epsilon), 0.0002f);
    }
}

void training() {
    Model model;
    model.add_layer<Dense>(tensor({1, 1}, {0}), tensor({1, 1}, {0}));
    auto x = tensor({1, 3}, {-1, 0, 1});
    auto y = tensor({1, 3}, {-1, 1, 3});
    MSELoss mse;
    for(int i = 0; i < 150; ++i) {
        auto prediction = model.forward(x);
        model.backward(mse.backward(prediction, y));
        model.update(0.1f);
    }
    equal(model.forward(x), y, 0.0001f);
    Model sequence;
    sequence.add_layer<Dense>(tensor({1, 1}, {2}), tensor({1, 1}, {1}));
    sequence.add_layer<ReLU>();
    sequence.add_layer<Dense>(tensor({1, 1}, {3}), tensor({1, 1}, {0}));
    auto single = tensor({1, 1}, {1});
    values(sequence.forward(single), {9});
    sequence.backward(tensor({1, 1}, {1}));
    sequence.update(0.1f);
    // Backward must use the old weights; then both Dense layers update once.
    values(sequence.forward(single), {6.38f});
    Model empty;
    equal(empty.forward(x), x);
    empty.backward(x);
    empty.update(0.1f);
}

void utilities() {
    for(int i = 0; i < 100; ++i) {
        float value = randf();
        check(std::isfinite(value) && value >= 0 && value <= 1, "Random range");
    }
    std::ostringstream output;
    auto* original = std::cout.rdbuf(output.rdbuf());
    try { print_tensor(tensor({1, 2}, {1, 2})); }
    catch(...) { std::cout.rdbuf(original); throw; }
    std::cout.rdbuf(original);
    check(output.str() == "1, 2, \n", "Tensor printing");
    throws<RequiredRankException>([] { print_tensor(Tensor({1})); });
    check(std::string(EqualRankException().what()).size() > 0, "Rank diagnostic");
}

void nonlinear_training() {
    Model model;
    model.add_layer<Dense>(
        tensor({4, 2}, {0.4f, -0.7f, 0.8f, 0.2f, -0.3f, 0.6f, -0.5f, -0.4f}),
        tensor({4, 1}, {0.1f, -0.2f, 0.3f, -0.1f}));
    model.add_layer<Tanh>();
    model.add_layer<Dense>(tensor({1, 4}, {0.4f, -0.2f, 0.3f, -0.5f}),
                           tensor({1, 1}, {0}));
    // XOR is not representable by a single affine layer.
    const auto input = tensor({2, 4}, {-1, -1, 1, 1, -1, 1, -1, 1});
    const auto target = tensor({1, 4}, {-1, 1, 1, -1});
    MSELoss loss;
    const float initial_loss = loss.forward(model.forward(input), target);
    for(int step = 0; step < 3000; ++step) {
        const auto prediction = model.forward(input);
        const float current_loss = loss.forward(prediction, target);
        check(std::isfinite(current_loss), "Nonlinear training diverged");
        model.backward(loss.backward(prediction, target));
        model.update(0.1f);
    }
    const auto prediction = model.forward(input);
    const float final_loss = loss.forward(prediction, target);
    check(final_loss < 0.001f && final_loss < initial_loss * 0.01f,
          "Nonlinear model failed to converge: " + std::to_string(final_loss));
    equal(prediction, target, 0.05f);
}

int main(int argc, char** argv) {
    return run_test([&] {
        check(argc == 2, "Expected case name");
        std::string name = argv[1];
        if(name == "dense") dense();
        else if(name == "activations") activations();
        else if(name == "loss") loss();
        else if(name == "gradients") gradients();
        else if(name == "training") training();
        else if(name == "nonlinear_training") nonlinear_training();
        else if(name == "utilities") utilities();
        else throw std::runtime_error("Unknown case");
    });
}
