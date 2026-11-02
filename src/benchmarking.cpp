#include "benchmarking.hpp"
#include "exceptions.hpp"
#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>

void Benchmark::start(std::string_view name) {
    if(started)
        throw std::logic_error("Nested benchmarks aren't allowed");
    temp.name = name;
    temp_now = clock.now();
    started = true;
}

void Benchmark::end() {
    if(!started)
        throw std::logic_error("Cannot end benchmark that haven't started yet");

    temp.duration = clock.now() - temp_now;
    temp_now = std::chrono::steady_clock::time_point::min();
    try {
        results.push_back(temp);
    } catch (std::exception& e){
        temp.clear();
        started = false;
        throw;
    }

    temp.clear();
    started = false;
}

Results Benchmark::get_results() {
    if(started)
        throw std::logic_error("Can't retrieve results in the middle of the benchmark");
    return std::exchange(results, {});
}
