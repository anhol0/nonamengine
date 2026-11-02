#pragma once

#include "benchmarking.hpp"

#include <cstddef>
#include <span>
#include <string_view>

namespace benchmarking {

struct Options {
    std::size_t warmup = 2;
    std::size_t samples = 7;
    std::size_t iterations = 1;
};

struct Statistics {
    double min_us;
    double median_us;
    double max_us;
};

void validate(const Options& options);
Statistics summarize(std::span<const BenchmarkResult> samples, std::size_t iterations);
double gigaflops(double operations, double microseconds);

// Defined in a separate compilation unit, with IPO disabled for the executable.
// Making each full result escape prevents dead-work elimination in Release
// builds without putting checksum calculations inside the timed interval.
void observe(std::span<const float> values) noexcept;

template<class Operation>
Statistics measure(std::string_view name, const Options& options, Operation operation) {
    validate(options);
    for(std::size_t batch = 0; batch < options.warmup; ++batch) {
        for(std::size_t i = 0; i < options.iterations; ++i) operation();
    }
    Benchmark timer;
    for(std::size_t sample = 0; sample < options.samples; ++sample) {
        timer.start(name);
        for(std::size_t i = 0; i < options.iterations; ++i) operation();
        timer.end();
    }
    const auto results = timer.get_results();
    return summarize(results, options.iterations);
}

} // namespace benchmarking
