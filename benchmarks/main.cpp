#include "runner.hpp"
#include "model.hpp"
#include "nn.hpp"
#include "tensor.hpp"

#include <CLI/CLI.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void fill_input(Tensor& tensor, float offset = 0.0f) {
    std::size_t index = 0;
    tensor.fill_with([&] {
        return offset + static_cast<float>(static_cast<int>(index++ % 29) - 14) / 32.0f;
    });
}

double checksum(const Tensor& tensor) {
    double total = 0;
    for(float value : tensor.data()) {
        if(!std::isfinite(value)) throw std::runtime_error("Non-finite benchmark result");
        total += value;
    }
    return total;
}

void report(std::string_view name, const std::string& shape,
            const benchmarking::Options& options, const benchmarking::Statistics& stats,
            double result_checksum, double operations = 0) {
    std::cout << name << ',' << shape << ',' << options.warmup << ',' << options.samples
              << ',' << options.iterations << ',' << stats.min_us << ',' << stats.median_us
              << ',' << stats.max_us << ',';
    if(operations > 0 && stats.median_us > 0) {
        std::cout << benchmarking::gigaflops(operations, stats.median_us);
    }
    std::cout << ',' << result_checksum << '\n';
}

bool selected(std::string_view name, const std::string& filter) {
    return name.find(filter) != std::string_view::npos;
}

std::size_t tensor_workloads(const benchmarking::Options& options, bool quick,
                             const std::string& filter) {
    const std::vector<std::array<uint64_t, 2>> shapes = quick ?
        std::vector<std::array<uint64_t, 2>>{{16, 16}, {8, 24}} :
        std::vector<std::array<uint64_t, 2>>{{64, 64}, {256, 256}, {1024, 1024}, {32, 1000}};
    std::size_t count = 0;
    for(const auto& [rows, columns] : shapes) {
        Tensor a({rows, columns}), b({rows, columns}), column({rows, 1}), row({1, columns});
        Tensor output({rows, columns});
        fill_input(a);
        fill_input(b, 1.0f); // Division has no zero denominators.
        fill_input(column);
        fill_input(row);
        const std::string shape = std::to_string(rows) + "x" + std::to_string(columns);
        auto run = [&](std::string_view name, auto operation) {
            if(!selected(name, filter)) return;
            const auto stats = benchmarking::measure(name, options, [&] {
                output = operation();
                benchmarking::observe(output.data());
            });
            report(name, shape, options, stats, checksum(output));
            ++count;
        };
        run("add", [&] { return a + b; });
        run("subtract", [&] { return a - b; });
        run("multiply", [&] { return a * b; });
        run("divide", [&] { return a / b; });
        run("add_scalar", [&] { return a + 2.0f; });
        run("subtract_scalar", [&] { return a - 2.0f; });
        run("multiply_scalar", [&] { return a * 2.0f; });
        run("divide_scalar", [&] { return a / 2.0f; });
        run("bias_column", [&] { return a + column; });
        run("bias_row", [&] { return a + row; });
        run("transpose", [&] { return a.transposed(); });
        run("relu", [&] { return tensor_operations::relu(a); });
    }
    return count;
}

std::size_t matmul_workloads(const benchmarking::Options& options, bool quick,
                             const std::string& filter) {
    if(!selected("matmul", filter)) return 0;
    // M, K, N: include forward and weight-gradient shapes from the demo network.
    const std::vector<std::array<uint64_t, 3>> shapes = quick ?
        std::vector<std::array<uint64_t, 3>>{{4, 3, 5}, {16, 16, 16}} :
        std::vector<std::array<uint64_t, 3>>{{32, 2, 1000}, {1, 32, 1000},
            {32, 1000, 2}, {1, 1000, 32}, {64, 64, 64}, {128, 128, 128}, {256, 256, 256}};
    for(const auto& [m, k, n] : shapes) {
        Tensor a({m, k}), b({k, n}), output({m, n});
        fill_input(a);
        fill_input(b);
        const auto stats = benchmarking::measure("matmul", options, [&] {
            output = a.matmul(b);
            benchmarking::observe(output.data());
        });
        const auto shape = std::to_string(m) + "x" + std::to_string(k) + "*" +
                           std::to_string(k) + "x" + std::to_string(n);
        report("matmul", shape, options, stats, checksum(output),
               2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n));
    }
    return shapes.size();
}

std::size_t training_workloads(const benchmarking::Options& options, bool quick,
                               const std::string& filter) {
    if(!selected("training_step", filter)) return 0;
    // Input features, hidden features, output features, batch size.
    const std::vector<std::array<uint64_t, 4>> shapes = quick ?
        std::vector<std::array<uint64_t, 4>>{{2, 8, 1, 8}} :
        std::vector<std::array<uint64_t, 4>>{{2, 32, 1, 1000}, {32, 64, 16, 64}};
    for(const auto& [inputs, hidden, outputs, batch] : shapes) {
        Model model;
        auto add_dense = [&](uint64_t in, uint64_t out) {
            Tensor weights({out, in}), biases({out, 1});
            fill_input(weights);
            weights = weights * (1.0f / std::sqrt(static_cast<float>(in)));
            biases.fill(0.01f);
            model.add_layer<Dense>(std::move(weights), std::move(biases));
        };
        add_dense(inputs, hidden);
        model.add_layer<ReLU>();
        add_dense(hidden, outputs);
        Tensor input({inputs, batch}), target({outputs, batch}), prediction({outputs, batch});
        fill_input(input);
        fill_input(target);
        MSELoss loss;
        float current_loss = 0;
        const auto stats = benchmarking::measure("training_step", options, [&] {
            prediction = model.forward(input);
            current_loss = loss.forward(prediction, target);
            const auto gradient = loss.backward(prediction, target);
            model.backward(gradient);
            model.update(0.01f);
            benchmarking::observe(prediction.data());
        });
        if(!std::isfinite(current_loss)) throw std::runtime_error("Non-finite training loss");
        const auto shape = std::to_string(inputs) + "->" + std::to_string(hidden) + "->" +
                           std::to_string(outputs) + "/batch=" + std::to_string(batch);
        report("training_step", shape, options, stats, checksum(prediction) + current_loss);
    }
    return shapes.size();
}

} // namespace

int main(int argc, char** argv) {
    CLI::App app{"CPU tensor and training benchmarks; use a Release build for comparisons"};
    benchmarking::Options options;
    bool quick = false;
    std::string filter;
    app.add_flag("--quick", quick, "Use small workloads for smoke checks");
    app.add_option("--warmup", options.warmup, "Untimed batches; each repeats --iterations calls")
        ->check(CLI::NonNegativeNumber)->capture_default_str();
    app.add_option("--samples", options.samples, "Timed batches")
        ->check(CLI::PositiveNumber)->capture_default_str();
    app.add_option("--iterations", options.iterations, "Calls per batch; report time per call")
        ->check(CLI::PositiveNumber)->capture_default_str();
    app.add_option("--filter", filter, "Select operation names containing this text");
    try {
        app.parse(argc, argv);
    } catch(const CLI::ParseError& error) {
        return app.exit(error);
    }
    try {
        benchmarking::validate(options);
        std::cout << "# compiler: " << NONAMENGINE_COMPILER << '\n'
                  << "# build: " << NONAMENGINE_BUILD_TYPE << '\n'
                  << "# time: microseconds per tensor operation or full training step\n"
                  << "# scope: public API allocation and validation included\n"
                  << "# profile: " << (quick ? "quick" : "standard") << '\n'
                  << "# inputs: fixed; training updates continue across warm-up and timed batches\n"
                  << "# checksum and reporting are outside timing; observer call overhead is included\n";
        std::cout << "name,shape,warmup,samples,iterations,min_us,median_us,max_us,gflops,checksum\n"
                  << std::fixed << std::setprecision(6);
        auto count = tensor_workloads(options, quick, filter);
        count += matmul_workloads(options, quick, filter);
        count += training_workloads(options, quick, filter);
        if(count == 0) throw std::invalid_argument("No workloads match --filter");
    } catch(const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
