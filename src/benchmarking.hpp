#pragma once

#include <chrono>
#include <string>
#include <string_view>
#include <vector>

class BenchmarkResult {
public:
    std::chrono::steady_clock::duration duration {};
    std::string name;
    void clear() {
        name.clear();
        duration = std::chrono::steady_clock::duration::zero();
    }
};

using Results = std::vector<BenchmarkResult>;

class Benchmark {
private:
    std::chrono::steady_clock clock;
    std::chrono::steady_clock::time_point temp_now;
    Results results;
    BenchmarkResult temp;
    bool started = false;
public:
    void start(std::string_view name);
    void end();
    Results get_results();
};
