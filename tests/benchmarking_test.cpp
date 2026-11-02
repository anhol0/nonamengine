#include "../src/benchmarking.hpp"
#include "test_support.hpp"

#include <array>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>

namespace {

using Clock = std::chrono::steady_clock;

void result_contract() {
    check((std::is_same_v<decltype(BenchmarkResult::duration), Clock::duration>),
          "BenchmarkResult::duration must store elapsed duration, not a time point");
    check((std::is_same_v<decltype(std::declval<Benchmark&>().get_results()), Results>),
          "get_results must return an owned Results vector by value");
}

void clear_result() {
    BenchmarkResult result {};
    result.name = "finished operation";
    result.duration = Clock::duration{123};
    result.clear();
    check(result.name.empty(), "clear must reset the result name");
    check(result.duration == decltype(result.duration){}, "clear must reset the elapsed duration to zero");
    result.clear();
    check(result.name.empty() && result.duration == decltype(result.duration){},
          "Clearing an already-cleared result must be harmless");
}

void empty_results() {
    Benchmark benchmark;
    check(benchmark.get_results().empty(), "A new benchmark must have no results");
    check(benchmark.get_results().empty(), "Repeated empty retrieval must stay empty");
}

void single_measurement() {
    Benchmark benchmark;
    benchmark.start("matmul");
    benchmark.end();
    const Results results = benchmark.get_results();
    check(results.size() == 1, "One interval must produce exactly one result");
    check(results.front().name == "matmul", "The interval name must be preserved");
    check(benchmark.get_results().empty(), "Retrieving results must drain the collection");
}

void ordered_results() {
    Benchmark benchmark;
    const std::array<std::string_view, 4> names {"add", "matmul", "add", "transpose"};
    for(const auto name : names) {
        benchmark.start(name);
        benchmark.end();
    }
    const Results results = benchmark.get_results();
    check(results.size() == names.size(), "Every interval must have its own result");
    for(std::size_t i = 0; i < names.size(); ++i) {
        check(results[i].name == names[i], "Preserve completion order and duplicate names");
    }
}

void repeated_batches() {
    Benchmark benchmark;
    for(int batch = 0; batch < 4; ++batch) {
        const auto name = "batch " + std::to_string(batch);
        benchmark.start(name);
        benchmark.end();
        const Results results = benchmark.get_results();
        check(results.size() == 1, "Previous batches must not accumulate after retrieval");
        check(results.front().name == name, "A fresh batch must record its own name");
        check(benchmark.get_results().empty(), "Each batch must be drained independently");
    }
}

void owned_names() {
    Benchmark benchmark;
    std::string name = "temporary operation name";
    const auto expected = name;
    benchmark.start(name);
    name.assign(name.size(), 'x');
    benchmark.end();
    {
        const std::string label = "prefix:transpose:suffix";
        benchmark.start(std::string_view(label).substr(7, 9));
    }
    benchmark.end();
    const Results results = benchmark.get_results();
    check(results.size() == 2, "Two named intervals must be retained");
    check(results[0].name == expected, "start must own its name before the caller changes it");
    check(results[1].name == "transpose", "Copy only the view's characters and own their lifetime");
}

void independent_results() {
    Results first;
    {
        Benchmark benchmark;
        benchmark.start("first");
        benchmark.end();
        first = benchmark.get_results();
        benchmark.start("second");
        benchmark.end();
        auto second = benchmark.get_results();
        check(first.size() == 1 && first.front().name == "first", "Later intervals must not mutate returned results");
        check(second.size() == 1 && second.front().name == "second", "Second retrieval must be independent");
        second.front().name = "edited";
        second.clear();
        check(benchmark.get_results().empty(), "Editing returned results must not repopulate the benchmark");
    }
    check(first.size() == 1 && first.front().name == "first", "Results must outlive the benchmark");
}

void independent_instances() {
    Benchmark first;
    Benchmark second;
    first.start("first timer");
    second.start("second timer");
    first.end();
    second.end();
    const auto first_results = first.get_results();
    const auto second_results = second.get_results();
    check(first_results.size() == 1 && first_results.front().name == "first timer", "First timer owns its state");
    check(second_results.size() == 1 && second_results.front().name == "second timer", "Second timer owns its state");
    check(first.get_results().empty() && second.get_results().empty(), "Both timers drain independently");
}

void many_measurements() {
    Benchmark benchmark;
    constexpr std::size_t count = 256;
    for(std::size_t i = 0; i < count; ++i) {
        benchmark.start("operation " + std::to_string(i));
        benchmark.end();
    }
    const auto results = benchmark.get_results();
    check(results.size() == count, "Growing the result vector must not lose measurements");
    for(std::size_t i = 0; i < count; ++i) {
        check(results[i].name == "operation " + std::to_string(i), "Names/order must survive vector growth");
    }
    check(benchmark.get_results().empty(), "A large batch must also drain completely");
}

void timing() {
    Benchmark benchmark;
    const auto before_start = Clock::now();
    benchmark.start("elapsed interval");
    const auto after_start = Clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto before_end = Clock::now();
    benchmark.end();
    const auto after_end = Clock::now();
    // Retrieval time is not part of the measured interval.
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto results = benchmark.get_results();
    check(results.size() == 1, "Timing must produce one result");
    const auto duration = results.front().duration;
    check(duration >= before_end - after_start, "Elapsed duration omitted work inside the interval");
    check(duration <= after_end - before_start, "Duration includes time outside the interval or uses incorrect units");
}

void independent_durations() {
    Benchmark benchmark;
    benchmark.start("earlier interval");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    benchmark.end();
    const auto before_start = Clock::now();
    benchmark.start("immediate interval");
    const auto after_start = Clock::now();
    const auto before_end = Clock::now();
    benchmark.end();
    const auto after_end = Clock::now();
    const auto results = benchmark.get_results();
    check(results.size() == 2, "Each timing interval must be retained");
    check(results[0].duration >= Clock::duration::zero(), "Duration cannot be negative");
    check(results[1].duration >= before_end - after_start, "The second interval ended too early");
    check(results[1].duration <= after_end - before_start, "Durations must not accumulate across intervals");
}

void end_without_start() {
    Benchmark benchmark;
    throws<std::logic_error>([&] { benchmark.end(); });
    check(benchmark.get_results().empty(), "Rejected end must not add a result");
    benchmark.start("recovery");
    benchmark.end();
    const auto results = benchmark.get_results();
    check(results.size() == 1 && results.front().name == "recovery", "A rejected end must leave the timer usable");
}

void double_end() {
    Benchmark benchmark;
    benchmark.start("completed");
    benchmark.end();
    throws<std::logic_error>([&] { benchmark.end(); });
    const auto results = benchmark.get_results();
    check(results.size() == 1 && results.front().name == "completed", "Rejected end must preserve exactly one completed result");
    throws<std::logic_error>([&] { benchmark.end(); });
    check(benchmark.get_results().empty(), "Draining must not reactivate the timer");
}

void nested_start() {
    Benchmark benchmark;
    benchmark.start("completed");
    benchmark.end();
    benchmark.start("outer");
    const auto after_start = Clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto before_nested_start = Clock::now();
    throws<std::logic_error>([&] { benchmark.start("inner"); });
    benchmark.end();
    const auto results = benchmark.get_results();
    check(results.size() == 2, "Rejected nested start must preserve completed and active intervals");
    check(results[0].name == "completed" && results[1].name == "outer", "Rejected nested start must not replace names");
    check(results[1].duration >= before_nested_start - after_start, "Rejected nested start must not reset the starting time");
}

void retrieval_while_active() {
    Benchmark benchmark;
    benchmark.start("first");
    throws<std::logic_error>([&] { (void)benchmark.get_results(); });
    benchmark.end();
    benchmark.start("second");
    const auto after_start = Clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto before_retrieval = Clock::now();
    throws<std::logic_error>([&] { (void)benchmark.get_results(); });
    benchmark.end();
    const auto results = benchmark.get_results();
    check(results.size() == 2, "Rejected retrieval must not discard completed results");
    check(results[0].name == "first" && results[1].name == "second", "Rejected retrieval must preserve the active interval");
    check(results[1].duration >= before_retrieval - after_start, "Rejected retrieval must not reset the starting time");
    check(benchmark.get_results().empty(), "Retrieval must work again after end");
}

} // namespace

int main(int argc, char** argv) {
    return run_test([&] {
        check(argc == 2, "Expected case name");
        const std::string name = argv[1];
        if(name == "result_contract") result_contract();
        else if(name == "clear") clear_result();
        else if(name == "empty") empty_results();
        else if(name == "single") single_measurement();
        else if(name == "order") ordered_results();
        else if(name == "batches") repeated_batches();
        else if(name == "names") owned_names();
        else if(name == "ownership") independent_results();
        else if(name == "instances") independent_instances();
        else if(name == "growth") many_measurements();
        else if(name == "timing") timing();
        else if(name == "durations") independent_durations();
        else if(name == "end_without_start") end_without_start();
        else if(name == "double_end") double_end();
        else if(name == "nested_start") nested_start();
        else if(name == "active_retrieval") retrieval_while_active();
        else throw std::runtime_error("Unknown case");
    });
}
