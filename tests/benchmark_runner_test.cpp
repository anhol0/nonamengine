#include "../benchmarks/runner.hpp"
#include "test_support.hpp"

#include <limits>

namespace {

Results samples(std::initializer_list<int> microseconds) {
    Results result;
    for(int time : microseconds) {
        result.push_back({std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::microseconds(time)), "sample"});
    }
    return result;
}

void statistics() {
    auto input = samples({90, 10, 50});
    auto stats = benchmarking::summarize(input, 10);
    near(stats.min_us, 1);
    near(stats.median_us, 5);
    near(stats.max_us, 9);
    check(input.front().duration == samples({90}).front().duration, "Statistics must not reorder input");
    stats = benchmarking::summarize(samples({80, 10, 60, 20}), 2);
    near(stats.min_us, 5);
    near(stats.median_us, 20);
    near(stats.max_us, 40);
    stats = benchmarking::summarize(samples({7}), 2);
    near(stats.min_us, 3.5f);
    near(stats.median_us, 3.5f);
    near(stats.max_us, 3.5f);
    stats = benchmarking::summarize(samples({0, 0}), 1);
    near(stats.median_us, 0);
}

void validation() {
    throws<std::invalid_argument>([] { (void)benchmarking::summarize({}, 1); });
    throws<std::invalid_argument>([] { (void)benchmarking::summarize(samples({1}), 0); });
    throws<std::invalid_argument>([] { (void)benchmarking::summarize(samples({-1}), 1); });
    int calls = 0;
    throws<std::invalid_argument>([&] {
        benchmarking::measure("bad samples", {0, 0, 1}, [&] { ++calls; });
    });
    throws<std::invalid_argument>([&] {
        benchmarking::measure("bad iterations", {1, 1, 0}, [&] { ++calls; });
    });
    check(calls == 0, "Invalid options must be rejected before executing work");
}

void repetitions() {
    int calls = 0;
    const auto stats = benchmarking::measure("counted", {2, 3, 4}, [&] { ++calls; });
    check(calls == 20, "Execute warm-up and sample batches with the requested iterations");
    check(stats.min_us >= 0 && stats.min_us <= stats.median_us && stats.median_us <= stats.max_us,
          "Measured statistics must be nonnegative and ordered");
    calls = 0;
    benchmarking::measure("no warm-up", {0, 2, 3}, [&] { ++calls; });
    check(calls == 6, "Zero warm-up must be supported");
}

void exceptions() {
    int calls = 0;
    throws<std::runtime_error>([&] {
        benchmarking::measure("warm-up failure", {1, 3, 2}, [&] {
            ++calls;
            throw std::runtime_error("operation failed");
        });
    });
    check(calls == 1, "Stop immediately on warm-up failure");
    calls = 0;
    throws<std::runtime_error>([&] {
        benchmarking::measure("measurement failure", {0, 3, 2}, [&] {
            ++calls;
            throw std::runtime_error("operation failed");
        });
    });
    check(calls == 1, "Stop immediately on measured-operation failure");
}

void throughput() {
    near(benchmarking::gigaflops(2e9, 1e6), 2);
    near(benchmarking::gigaflops(2e6, 1000), 2);
    for(double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        throws<std::invalid_argument>([&] { (void)benchmarking::gigaflops(1, invalid); });
        throws<std::invalid_argument>([&] { (void)benchmarking::gigaflops(invalid, 1); });
    }
}

} // namespace

int main(int argc, char** argv) {
    return run_test([&] {
        check(argc == 2, "Expected case name");
        const std::string name = argv[1];
        if(name == "statistics") statistics();
        else if(name == "validation") validation();
        else if(name == "repetitions") repetitions();
        else if(name == "exceptions") exceptions();
        else if(name == "throughput") throughput();
        else throw std::runtime_error("Unknown case");
    });
}
