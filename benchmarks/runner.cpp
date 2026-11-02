#include "runner.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <vector>

void benchmarking::validate(const Options& options) {
    if(options.samples == 0 || options.iterations == 0) {
        throw std::invalid_argument("Samples and iterations must be positive");
    }
}

benchmarking::Statistics benchmarking::summarize(
    std::span<const BenchmarkResult> samples, std::size_t iterations
) {
    if(samples.empty() || iterations == 0) {
        throw std::invalid_argument("Statistics require samples and positive iterations");
    }
    std::vector<double> times;
    times.reserve(samples.size());
    for(const auto& sample : samples) {
        if(sample.duration < std::chrono::steady_clock::duration::zero()) {
            throw std::invalid_argument("Elapsed durations cannot be negative");
        }
        times.push_back(std::chrono::duration<double, std::micro>(sample.duration).count() /
                        static_cast<double>(iterations));
    }
    std::sort(times.begin(), times.end());
    const auto middle = times.size() / 2;
    const double median = times.size() % 2 == 0 ?
        (times[middle - 1] + times[middle]) / 2 : times[middle];
    return {times.front(), median, times.back()};
}

double benchmarking::gigaflops(double operations, double microseconds) {
    if(!std::isfinite(operations) || !std::isfinite(microseconds) ||
       operations <= 0 || microseconds <= 0) {
        throw std::invalid_argument("Throughput requires positive finite work and time");
    }
    return operations / (microseconds * 1000.0);
}

void benchmarking::observe(std::span<const float>) noexcept {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}
